import random
import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 80.0  # seconds, safety margin before 90s kill
    
    n = 15
    all_possible = set((i, j) for i in range(n) for j in range(i+1, n))
    
    def is_valid(edges_set):
        """
        Check if the graph represented by edges_set is valid.
        Violation: Simple path s-u-v-t where d(s) == d(t).
        """
        # Build adjacency and degrees
        adj = [set() for _ in range(n)]
        degrees = [0] * n
        
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            
        # Group vertices by degree
        deg_groups = {}
        for i in range(n):
            deg_groups.setdefault(degrees[i], set()).add(i)
        
        # For each pair (s, t) with same degree, check if 3-path exists
        for deg, group in deg_groups.items():
            group_list = list(group)
            for idx_s in range(len(group_list)):
                s = group_list[idx_s]
                for idx_t in range(idx_s + 1, len(group_list)):
                    t = group_list[idx_t]
                    # Check if path s-u-v-t exists with s,u,v,t distinct
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[u]:
                            if v == s or v == t or v == u:
                                continue
                            if t in adj[v]:
                                return False
        return True

    def get_violating_pair(edges_set):
        """Find a violating (s,t) pair if any, for potential targeted repair."""
        adj = [set() for _ in range(n)]
        degrees = [0] * n
        
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            
        deg_groups = {}
        for i in range(n):
            deg_groups.setdefault(degrees[i], set()).add(i)
        
        for deg, group in deg_groups.items():
            group_list = list(group)
            for idx_s in range(len(group_list)):
                s = group_list[idx_s]
                for idx_t in range(idx_s + 1, len(group_list)):
                    t = group_list[idx_t]
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[u]:
                            if v == s or v == t or v == u:
                                continue
                            if t in adj[v]:
                                return (s, t)
        return None

    def local_search(initial_edges):
        edges = set(tuple(sorted(e)) for e in initial_edges)
        
        # Ensure validity first
        if not is_valid(edges):
            valid = False
            attempts = 0
            while not is_valid(edges) and attempts < 500:
                e_to_remove = random.choice(list(edges))
                edges.remove(e_to_remove)
                attempts += 1
            if not is_valid(edges):
                return None, 0

        current_score = len(edges)
        
        # Phase 1: Greedy addition with multiple passes
        improved = True
        pass_count = 0
        while improved and pass_count < 20:
            improved = False
            pass_count += 1
            missing = [e for e in all_possible if e not in edges]
            random.shuffle(missing)
            
            for e in missing:
                if time.time() - start_time > deadline:
                    return list(edges), current_score
                new_edges = edges | {e}
                if is_valid(new_edges):
                    edges = new_edges
                    current_score = len(edges)
                    improved = True
                    break
        
        # Phase 2: Swap moves (remove 1, add 1) to escape local optima
        swap_improved = True
        max_swap_iterations = 30
        swap_iter = 0
        while swap_improved and swap_iter < max_swap_iterations:
            swap_improved = False
            swap_iter += 1
            
            missing = [e for e in all_possible if e not in edges]
            random.shuffle(missing)
            existing = list(edges)
            random.shuffle(existing)
            
            for e_remove in existing:
                if time.time() - start_time > deadline:
                    return list(edges), current_score
                edges_without = edges - {e_remove}
                
                for e_add in missing:
                    if e_add == e_remove:
                        continue
                    new_edges = edges_without | {e_add}
                    if is_valid(new_edges):
                        edges = new_edges
                        current_score = len(edges)
                        swap_improved = True
                        break
                if swap_improved:
                    break
            
            # After a swap, try to add more edges greedily
            if swap_improved:
                add_improved = True
                while add_improved:
                    if time.time() - start_time > deadline:
                        return list(edges), current_score
                    add_improved = False
                    missing = [e for e in all_possible if e not in edges]
                    random.shuffle(missing)
                    for e in missing:
                        new_edges = edges | {e}
                        if is_valid(new_edges):
                            edges = new_edges
                            current_score = len(edges)
                            add_improved = True
                            break
        
        # Phase 3: 2-for-1 swaps (remove 2, add 1) for deeper exploration
        two_for_one_improved = True
        max_21_iter = 15
        two_21_iter = 0
        while two_for_one_improved and two_21_iter < max_21_iter:
            two_for_one_improved = False
            two_21_iter += 1
            
            missing = [e for e in all_possible if e not in edges]
            random.shuffle(missing)
            existing = list(edges)
            random.shuffle(existing)
            
            for i in range(min(len(existing), 50)):
                if time.time() - start_time > deadline:
                    return list(edges), current_score
                e_remove1 = existing[i]
                for j in range(i+1, min(len(existing), 50)):
                    e_remove2 = existing[j]
                    edges_without = edges - {e_remove1, e_remove2}
                    
                    for e_add in missing[:20]:  # Limit candidates for speed
                        if e_add == e_remove1 or e_add == e_remove2:
                            continue
                        new_edges = edges_without | {e_add}
                        if is_valid(new_edges):
                            edges = new_edges
                            current_score = len(edges)
                            two_for_one_improved = True
                            break
                    if two_for_one_improved:
                        break
                if two_for_one_improved:
                    break
        
        return list(edges), current_score

    def perturbed_local_search(best_edges, perturbation_size=3):
        """Start from best, remove some edges, then re-optimize."""
        edges = set(tuple(sorted(e)) for e in best_edges)
        # Remove some random edges
        to_remove = random.sample(list(edges), min(perturbation_size, len(edges)))
        for e in to_remove:
            edges.remove(e)
        
        return local_search(list(edges))

    best_edges = None
    best_score = -1

    # Generate diverse initial seeds
    all_edges_list = list(all_possible)
    
    # Strategy 1: Random dense starts
    num_random_starts = 30
    for i in range(num_random_starts):
        if time.time() - start_time > deadline:
            break
        
        if i < num_random_starts // 3:
            density = 0.65 + 0.3 * random.random()
        elif i < 2 * num_random_starts // 3:
            density = 0.5 + 0.4 * random.random()
        else:
            density = 0.35 + 0.45 * random.random()
        
        num_initial = int(len(all_edges_list) * density)
        initial_edges = random.sample(all_edges_list, num_initial)
        
        res_edges, score = local_search(initial_edges)
        
        if res_edges is not None and score > best_score:
            best_score = score
            best_edges = res_edges
    
    # Strategy 2: Structured starts - near-regular graphs
    # Try to build graphs with more uniform degrees
    for trial in range(15):
        if time.time() - start_time > deadline:
            break
        
        # Start with a cycle or random regular-ish structure
        if trial % 3 == 0:
            # Start with a Hamiltonian cycle
            perm = list(range(n))
            random.shuffle(perm)
            initial_edges = []
            for i in range(n):
                u = perm[i]
                v = perm[(i + 1) % n]
                initial_edges.append((min(u,v), max(u,v)))
            # Add some random chords
            extra = random.sample(all_edges_list, 25)
            for e in extra:
                if e not in set(tuple(sorted(x)) for x in initial_edges):
                    initial_edges.append(e)
        elif trial % 3 == 1:
            # Start with random near-regular: each vertex gets ~5-7 edges
            initial_edges = set()
            for i in range(n):
                target_deg = random.randint(5, 7)
                candidates = [j for j in range(n) if j != i and (min(i,j), max(i,j)) not in initial_edges]
                random.shuffle(candidates)
                for j in candidates:
                    if len([e for e in initial_edges if i in e]) >= target_deg:
                        break
                    initial_edges.add((min(i,j), max(i,j)))
            initial_edges = list(initial_edges)
        else:
            # Start with two disjoint cycles plus random connections
            initial_edges = set()
            perm1 = list(range(7))
            random.shuffle(perm1)
            for i in range(7):
                u = perm1[i]
                v = perm1[(i + 1) % 7]
                initial_edges.add((min(u,v), max(u,v)))
            perm2 = list(range(7, 15))
            random.shuffle(perm2)
            for i in range(8):
                u = perm2[i]
                v = perm2[(i + 1) % 8]
                initial_edges.add((min(u,v), max(u,v)))
            # Add random cross-edges
            cross_candidates = [(u, v) for u in range(7) for v in range(7, 15) 
                               if (min(u,v), max(u,v)) not in initial_edges]
            random.shuffle(cross_candidates)
            for e in cross_candidates[:15]:
                initial_edges.add(e)
            initial_edges = list(initial_edges)
        
        res_edges, score = local_search(initial_edges)
        
        if res_edges is not None and score > best_score:
            best_score = score
            best_edges = res_edges
    
    # Strategy 3: Perturbation from best found
    if best_edges is not None:
        num_perturbations = 30
        for i in range(num_perturbations):
            if time.time() - start_time > deadline:
                break
            
            pert_size = random.randint(2, 6)
            res_edges, score = perturbed_local_search(best_edges, pert_size)
            
            if res_edges is not None and score > best_score:
                best_score = score
                best_edges = res_edges
    
    # Final: try to squeeze more from best with multiple passes
    if best_edges is not None:
        for _ in range(3):
            if time.time() - start_time > deadline:
                break
            res_edges, score = local_search(best_edges)
            if res_edges is not None and score > best_score:
                best_score = score
                best_edges = res_edges
            else:
                break

    if best_edges is None:
        # Fallback: Empty graph is valid
        return []
        
    # Convert to required format: list of lists [u, v]
    # Ensure all values are Python int (not numpy int or bool)
    result = [[int(u), int(v)] for u, v in best_edges]
    return result

if __name__ == "__main__":
    pass
