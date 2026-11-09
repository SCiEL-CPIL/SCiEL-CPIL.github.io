import time
import itertools
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin
    
    n = 15
    # Partition: A = 0..6 (size 7), B = 7..14 (size 8)
    A = list(range(7))
    B = list(range(7, 15))
    
    # Start with complete bipartite K_{7,8}
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add((min(a,b), max(a,b)))
    
    # Candidate intra-partition edges
    candidates = []
    for i, j in itertools.combinations(A, 2):
        candidates.append((i, j))
    for i, j in itertools.combinations(B, 2):
        candidates.append((i, j))
    
    def get_degrees(edge_set):
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def is_valid(edge_set):
        deg = get_degrees(edge_set)
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        # Check every 3-edge path s-u-v-t
        for u, v in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    def count_conflicts(edge_set, candidate_e):
        """Count how many existing edges conflict with adding candidate_e"""
        e = (min(candidate_e[0], candidate_e[1]), max(candidate_e[0], candidate_e[1]))
        if e in edge_set:
            return 0
        test_set = set(edge_set)
        test_set.add(e)
        # Check if adding this edge creates any violation
        if is_valid(test_set):
            return 0
        # Count which existing edges are involved in violations
        deg = get_degrees(test_set)
        adj = [set() for _ in range(n)]
        for u, v in test_set:
            adj[u].add(v)
            adj[v].add(u)
        conflict_count = 0
        for u, v in edge_set:
            # Check if removing this edge would fix the issue (simplified: just count total)
            pass
        return 1  # Just indicate it's invalid
    
    best_edges = set(base_edges)
    best_count = len(base_edges)
    
    # Try multiple random orderings
    for seed in range(50):
        if time.time() > DEADLINE:
            break
        
        rng = random.Random(seed)
        cand_list = list(candidates)
        rng.shuffle(cand_list)
        
        current = set(base_edges)
        
        # Priority-based insertion: sort by "safety" (fewer conflicts)
        # We'll do greedy with backtracking-lite
        for i, j in cand_list:
            if time.time() > DEADLINE:
                break
            e = (min(i, j), max(i, j))
            if e in current:
                continue
            current.add(e)
            if is_valid(current):
                pass
            else:
                current.discard(e)
        
        if len(current) > best_count:
            best_edges = set(current)
            best_count = len(current)
        
        # Local search: 1-swap (remove 1, add 1)
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            # Try all 1-swaps
            current_list = list(current)
            missing_list = list(all_edges - current)
            
            # Shuffle to avoid bias
            rng.shuffle(current_list)
            rng.shuffle(missing_list)
            
            for e1 in current_list:
                if time.time() > DEADLINE:
                    break
                for e2 in missing_list:
                    if time.time() > DEADLINE:
                        break
                    if e1 == e2:
                        continue
                    new_set = set(current)
                    new_set.discard(e1)
                    new_set.add(e2)
                    
                    if is_valid(new_set):
                        # Accept if it increases edge count (it always does if e1!=e2 and e2 not in current)
                        # Wait, len(new_set) == len(current).
                        # We are swapping. Size is constant.
                        # We need a different objective or accept swaps to move in search space.
                        # But we want MAX edges.
                        # If we are stuck at size K, and no single add works,
                        # a swap keeps size K. It might allow a subsequent add.
                        # So we should accept swaps that lead to a "better" state?
                        # Hard to define better.
                        # Let's accept any valid swap with some probability, or just move.
                        # To avoid loops, we can track visited states? Too expensive.
                        # Let's just accept the first valid swap found, then restart.
                        current = new_set
                        improved = True
                        break
                if improved:
                    break
            
            # Try pure Add again
            for e2 in missing_list:
                if time.time() > DEADLINE:
                    break
                new_set = set(current)
                new_set.add(e2)
                if is_valid(new_set):
                    current = new_set
                    improved = True
                    break
            
            if len(current) > best_count:
                best_edges = set(current)
                best_count = len(current)
    
    # Final check and convert
    if not is_valid(best_edges):
        # Fallback to base if somehow invalid
        best_edges = set(base_edges)
        
    result = [[u, v] for u, v in sorted(best_edges)]
    return result
