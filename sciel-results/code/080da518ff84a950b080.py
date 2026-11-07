import random
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # seconds, safety margin before 90s kill

    n = 15
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_edges_set = set(all_edges)
    num_edges_total = len(all_edges)  # 105

    def is_valid(edge_set):
        """
        Check if the graph represented by edge_set is valid.
        Violation: Simple path s-u-v-t where d(s) == d(t).
        """
        if not edge_set:
            return True

        adj = [set() for _ in range(n)]
        degrees = [0] * n

        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1

        # Group vertices by degree
        deg_groups = {}
        for i in range(n):
            deg_groups.setdefault(degrees[i], set()).add(i)

        # For each group of vertices with the same degree, check if any pair
        # is connected by a path of length 3 (s-u-v-t)
        for deg, group in deg_groups.items():
            group_list = list(group)
            m = len(group_list)
            if m < 2:
                continue

            # For each pair (s, t) in the group, check if there's a path s-u-v-t
            # A path s-u-v-t exists if:
            #   u in N(s), v in N(u) \ {s}, t in N(v) \ {u, s}
            #   and s, u, v, t are all distinct
            
            # Optimization: Precompute for each s in group, the set of vertices
            # reachable by a path of length 3 from s.
            # Actually, let's just check directly.
            
            for i in range(m):
                s = group_list[i]
                for j in range(i + 1, m):
                    t = group_list[j]
                    # Check if path s-u-v-t exists
                    found = False
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[u]:
                            if v == s or v == t:
                                continue
                            if t in adj[v]:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return False
        return True

    def try_add_edges(edge_set, time_limit):
        """
        Try to add edges one by one to the given edge_set.
        Returns the new edge_set.
        """
        current = set(edge_set)
        # Try adding edges in a random order
        missing = [e for e in all_edges if e not in current]
        random.shuffle(missing)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Shuffle again if we made progress? Or just continue?
            # Let's continue with the current shuffled list, but if we pass through
            # without improvement, break.
            for e in missing:
                if e in current:
                    continue
                new_set = current | {e}
                if is_valid(new_set):
                    current = new_set
                    improved = True
                    # We can't remove e from missing easily, but we check `if e in current`
                    # so it will be skipped next time.
                    # However, adding an edge might make other previously invalid edges valid?
                    # No, adding edges only increases degrees and creates more paths.
                    # It cannot make an invalid graph valid.
                    # So we don't need to re-check previously rejected edges.
                    # Wait, is that true?
                    # If G is invalid, adding an edge makes it more invalid?
                    # The constraint is "No path s-u-v-t with d(s)==d(t)".
                    # Adding an edge (u,v) increases d(u) and d(v).
                    # It might change d(s) or d(t) so they are no longer equal.
                    # So a previously invalid pair might become valid!
                    # Example: s and t both have degree 2. Path exists. Invalid.
                    # Add edge (s, x). d(s)=3, d(t)=2. Not equal. Valid for this pair.
                    # So we DO need to re-check.
                    # Therefore, simple greedy addition in one pass is not sufficient.
                    # We need to iterate until no more edges can be added.
                    # My loop `while improved` handles this.
                    # But `missing` list is fixed. If I add an edge, I should re-shuffle
                    # or just continue?
                    # If I continue, I might miss edges that became valid later.
                    # So I should restart the scan or re-shuffle.
                    # Let's break and restart to be safe.
                    break
        
        return current

    def local_search(start_set, time_limit):
        """
        Perform local search starting from start_set.
        Uses add, remove, and swap moves.
        """
        current = set(start_set)
        if not is_valid(current):
            # Try to fix by removing edges
            # This is a heuristic fix
            edges_to_check = list(current)
            random.shuffle(edges_to_check)
            for e in edges_to_check:
                if time.time() > time_limit:
                    break
                trial = current - {e}
                if is_valid(trial):
                    current = trial
        
        if not is_valid(current):
            return current

        best = current
        best_size = len(current)
        
        # Try to improve by adding edges
        current = try_add_edges(current, time_limit)
        if len(current) > best_size:
            best = current
            best_size = len(current)

        # Now try swaps: remove 1, add 1
        # Or remove 2, add 1?
        # Let's try remove 1, add 1 first.
        # We iterate until no improvement or time limit.
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Generate random moves
            # To be systematic, we can iterate over all edges
            # But 105 edges is small.
            
            # Strategy:
            # 1. Try adding any missing edge (already done in try_add_edges, but maybe
            #    new ones became possible? No, try_add_edges loops until stable).
            #    So current is maximal w.r.t additions.
            
            # 2. Try swapping: remove e_in, add e_out
            #    We want to increase size, so we need to remove 1 and add 2?
            #    Or remove 2 and add 1? That decreases size.
            #    So we are looking for a sequence of moves that increases size.
            #    Since we are stuck on additions, we must remove something to add something else.
            #    But removing an edge decreases size by 1. Adding one increases by 1. Net 0.
            #    So we need a cycle of swaps that eventually allows an addition.
            #    This is complex.
            
            # Simpler approach:
            # Try removing an edge, then see if we can add MORE than 1 edge.
            # i.e., Remove e. Then run try_add_edges.
            # If len(new) > len(old), accept.
            
            edges_in_current = list(current)
            random.shuffle(edges_in_current)
            
            for e_remove in edges_in_current:
                if time.time() > time_limit:
                    break
                trial = current - {e_remove}
                # Try to fill up
                trial_expanded = try_add_edges(trial, time_limit)
                if len(trial_expanded) > best_size:
                    current = trial_expanded
                    best = current
                    best_size = len(current)
                    improved = True
                    break # Restart loop with new current
            
        return best

    best_edges = set()
    
    # Multiple restarts
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial set
        # Random subset
        size = random.randint(0, 30)
        initial = set(random.sample(all_edges, size))
        
        # Ensure valid start
        if not is_valid(initial):
            # Fix it
            edges_to_check = list(initial)
            random.shuffle(edges_to_check)
            for e in edges_to_check:
                trial = initial - {e}
                if is_valid(trial):
                    initial = trial
        
        if not is_valid(initial):
            continue

        # Local search
        result = local_search(initial, deadline)
        
        if len(result) > len(best_edges):
            best_edges = result
            # If we found a very good solution, we might stop early?
            # Max is 105.
            if len(best_edges) > 80:
                # Try a few more to see if we can push higher?
                # Or break?
                # Let's keep going for a bit more if we have time.
                pass

    # Final polish on best_edges
    if time.time() < deadline - 5:
        best_edges = try_add_edges(best_edges, deadline - 5)
        # One more round of local search on the best
        if time.time() < deadline - 2:
            best_edges = local_search(best_edges, deadline - 2)
            best_edges = try_add_edges(best_edges, deadline)

    # Convert to required format
    # Ensure types are int
    result_list = []
    for u, v in best_edges:
        result_list.append([int(u), int(v)])
        
    # Ensure sorted for consistency? Not required but good.
    # result_list.sort()
    
    return result_list

if __name__ == "__main__":
    pass
