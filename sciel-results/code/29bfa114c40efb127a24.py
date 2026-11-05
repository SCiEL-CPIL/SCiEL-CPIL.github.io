import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    def get_paley(q):
        """
        Returns adjacency list (as list of sets) for Paley graph of order q.
        Vertices are 0..q-1.
        """
        if q % 4 != 1:
            return None
        residues = {(x * x) % q for x in range(1, q)}
        adj = [set() for _ in range(q)]
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    def find_violation(n, adj):
        """
        Finds a K5 or an independent set of size 5.
        Returns (vertices, is_clique) or (None, None) if none found.
        """
        # Check for K5
        # Optimization: Iterate over triangles? Or just 5-combos?
        # For n=41, 5-combos is ~750k. 
        # Checking a 5-combo is O(1) with sets.
        # 750k * 10 ops ~ 7.5M ops. Should be fast enough in Py if optimized?
        # Let's try.
        
        # To speed up, we can pre-compute common neighbors.
        # But let's start with simple combo check.
        
        for combo in itertools.combinations(range(n), 5):
            is_k5 = True
            is_ik5 = True
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = combo[i], combo[j]
                    if v in adj[u]:
                        is_ik5 = False
                    else:
                        is_k5 = False
                    if not is_k5 and not is_ik5:
                        break
                if not is_k5 and not is_ik5:
                    break
            if is_k5:
                return combo, True
            if is_ik5:
                return combo, False
        
        return None, None

    def is_valid(n, adj):
        v, _ = find_violation(n, adj)
        return v is None

    def local_search(n, adj, time_limit):
        """
        Simulated annealing / local search to remove violations.
        """
        current_adj = [set(s) for s in adj]
        current_violation, current_type = find_violation(n, current_adj)
        
        # If already valid, return
        if current_violation is None:
            return current_adj
        
        temp = 1.0
        cooling = 0.999
        min_temp = 0.01
        
        while time.time() < time_limit and temp > min_temp:
            # Find a violation
            violation, is_clique = find_violation(n, current_adj)
            if violation is None:
                return current_adj
            
            # Pick a random edge in the violation to flip
            # If it's a clique, we want to remove an edge.
            # If it's an independent set, we want to add an edge.
            # But flipping an edge in a clique might break the clique but create a new one elsewhere.
            # We just pick a random pair in the violation.
            
            u, v = random.sample(list(violation), 2)
            
            # Flip edge
            if v in current_adj[u]:
                current_adj[u].remove(v)
                current_adj[v].remove(u)
            else:
                current_adj[u].add(v)
                current_adj[v].add(u)
                
            # Check new violation
            new_violation, new_type = find_violation(n, current_adj)
            
            # Simple acceptance: if we fixed the specific violation, accept.
            # But we don't easily know if we fixed it without re-checking the specific set.
            # Let's just accept if the *count* of violations decreased? 
            # Counting is expensive.
            
            # Heuristic: Accept if the specific violation is gone.
            # We can check if 'violation' is still a violation.
            is_still_violation = True
            if is_clique:
                for i in range(5):
                    for j in range(i+1, 5):
                        if violation[j] not in current_adj[violation[i]]:
                            is_still_violation = False
                            break
                    if not is_still_violation:
                        break
            else:
                for i in range(5):
                    for j in range(i+1, 5):
                        if violation[j] in current_adj[violation[i]]:
                            is_still_violation = False
                            break
                    if not is_still_violation:
                        break
            
            if not is_still_violation:
                # We fixed this one. Accept.
                pass
            else:
                # We didn't fix it (or created a different one). 
                # Revert? Or accept with prob?
                # To avoid oscillation, let's revert if we didn't fix the target violation.
                if v in current_adj[u]:
                    current_adj[u].remove(v)
                    current_adj[v].remove(u)
                else:
                    current_adj[u].add(v)
                    current_adj[v].add(u)
                
                # If we can't fix it easily, maybe try a different edge?
                # For simplicity, we just continue the loop, which will pick a new violation.
                # But if we keep failing, we might get stuck.
                # Let's try a random flip regardless if stuck?
                # Actually, the "revert" logic above means we try a different edge next time.
                # But we picked a random edge. If we revert, we are back to start.
                # We need to try *all* edges in the violation?
                # That's 10 edges.
                
                # Let's refine: Try all edges in the violation. If any fixes it, take it.
                # If none fixes it, revert and try a different violation or random move.
                
                # Re-implementing the inner loop to try all edges in violation:
                fixed = False
                for u2, v2 in itertools.combinations(violation, 2):
                    # Flip
                    if v2 in current_adj[u2]:
                        current_adj[u2].remove(v2)
                        current_adj[v2].remove(u2)
                    else:
                        current_adj[u2].add(v2)
                        current_adj[v2].add(u2)
                    
                    # Check if 'violation' is fixed
                    is_fixed = True
                    if is_clique:
                        for i in range(5):
                            for j in range(i+1, 5):
                                if violation[j] not in current_adj[violation[i]]:
                                    is_fixed = False
                                    break
                            if not is_fixed: break
                    else:
                        for i in range(5):
                            for j in range(i+1, 5):
                                if violation[j] in current_adj[violation[i]]:
                                    is_fixed = False
                                    break
                            if not is_fixed: break
                    
                    if is_fixed:
                        fixed = True
                        break
                    else:
                        # Revert this specific flip
                        if v2 in current_adj[u2]:
                            current_adj[u2].remove(v2)
                            current_adj[v2].remove(u2)
                        else:
                            current_adj[u2].add(v2)
                            current_adj[v2].add(u2)
                
                if not fixed:
                    # No single edge flip in this violation fixes it.
                    # This can happen if the violation is part of a larger structure.
                    # In this case, we might need to break it by flipping an edge *not* in the violation?
                    # Or accept a "worse" state?
                    # For now, let's just break the loop if we can't make progress?
                    # Or try a random flip elsewhere?
                    pass

            temp *= cooling

        return current_adj

    # 1. Try Paley 41
    p41 = get_paley(41)
    if p41:
        if is_valid(41, p41):
            # Convert to edges
            edges = []
            for i in range(41):
                for j in p41[i]:
                    if i < j:
                        edges.append((i, j))
            return 41, edges

    # 2. Try Local Search on n=41 starting from Paley 41 (even if invalid)
    # Or start from random?
    # Paley 41 is a good start.
    if p41:
        t_limit = deadline - 30
        adj_ls = local_search(41, p41, t_limit)
        if is_valid(41, adj_ls):
            edges = []
            for i in range(41):
                for j in adj_ls[i]:
                    if i < j:
                        edges.append((i, j))
            return 41, edges

    # 3. Try Local Search on n=42?
    # This is harder. Maybe skip if time is tight.
    # Let's try to extend Paley 41 to 42?
    # Or just try random search on 42?
    # Given time constraints, 41 is a solid target.
    
    # 4. Fallback to Paley 37
    p37 = get_paley(37)
    if p37:
        edges = []
        for i in range(37):
            for j in p37[i]:
                if i < j:
                    edges.append((i, j))
        return 37, edges

    return 0, []
