import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    n = 16
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Precompute all 8-vertex subsets
    subsets_8 = list(itertools.combinations(range(n), 8))
    
    # Q3 structure: bipartite with even parity {0,3,5,6} and odd parity {1,2,4,7}
    # Edges in Q3 (as (even_idx, odd_idx) where even_idx in 0..3 maps to {0,3,5,6},
    # odd_idx in 0..3 maps to {1,2,4,7})
    even_positions = [0, 3, 5, 6]
    odd_positions = [1, 2, 4, 7]
    q3_edges = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,3), (2,0), (2,2), (2,3), (3,1), (3,2), (3,3)]
    q3_edges_set = set(q3_edges)
    
    # Precompute permutations of 4 elements
    perms_4 = list(itertools.permutations(range(4)))
    
    # Precompute all splits of {0..7} into two 4-sets (70 ways)
    splits = list(itertools.combinations(range(8), 4))
    
    def has_q3(adj):
        """
        adj: 16x16 numpy array of bools
        Returns True if the graph contains Q3 as a subgraph.
        """
        for s in subsets_8:
            # Quick edge count with early termination
            ec = 0
            for i in range(8):
                vi = s[i]
                for j in range(i+1, 8):
                    if adj[vi, s[j]]:
                        ec += 1
                        if ec >= 12:
                            break
                if ec >= 12:
                    break
            if ec < 12:
                continue
            
            # Detailed check: try all 70 splits
            for split in splits:
                even_set = [s[i] for i in split]
                even_set_set = set(split)
                odd_set = [s[i] for i in range(8) if i not in even_set_set]
                
                # Build 4x4 adjacency matrix between even_set and odd_set
                bipartite = [[adj[even_set[i], odd_set[j]] for j in range(4)] for i in range(4)]
                
                # Try all 24 permutations of the odd side (pe = identity)
                for po in perms_4:
                    ok = True
                    for e in range(4):
                        for o in range(4):
                            if (e, o) in q3_edges_set:
                                if not bipartite[e][po[o]]:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok:
                        return True
                
                # Try all 24 permutations of the even side (po = identity)
                for pe in perms_4:
                    ok = True
                    for e in range(4):
                        for o in range(4):
                            if (e, o) in q3_edges_set:
                                if not bipartite[pe[e]][o]:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok:
                        return True
        return False
    
    import numpy as np
    
    # Start with a known dense Q3-free construction
    # Use a random graph and remove Q3 subgraphs
    random.seed(42)
    
    def make_adj(edge_set):
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edge_set:
            adj[u, v] = True
            adj[v, u] = True
        return adj
    
    # Initial graph: random with p=0.5
    current_edges = set()
    for i, j in all_pairs:
        if random.random() < 0.5:
            current_edges.add((i, j))
    
    adj = make_adj(current_edges)
    
    # Remove any Q3 subgraphs
    changed = True
    while changed and time.time() < deadline - 30:
        changed = False
        if has_q3(adj):
            edge_list = list(current_edges)
            if edge_list:
                rem_u, rem_v = random.choice(edge_list)
                adj[rem_u, rem_v] = False
                adj[rem_v, rem_u] = False
                current_edges.remove((rem_u, rem_v))
                changed = True
    
    best_edges = set(current_edges)
    best_count = len(current_edges)
    
    # Local search: try adding edges
    non_edges = [e for e in all_pairs if e not in current_edges]
    
    for restart in range(10):
        if time.time() > deadline - 15:
            break
        
        # Work from current best
        adj = make_adj(best_edges)
        current_edges = set(best_edges)
        
        non_edges = [e for e in all_pairs if e not in current_edges]
        random.shuffle(non_edges)
        
        # Greedy addition
        for u, v in non_edges:
            if time.time() > deadline - 5:
                break
            if (u, v) in current_edges:
                continue
            adj[u, v] = True
            adj[v, u] = True
            current_edges.add((u, v))
            
            if has_q3(adj):
                adj[u, v] = False
                adj[v, u] = False
                current_edges.remove((u, v))
            else:
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = set(current_edges)
        
        # 2-opt: remove one edge, add up to two
        if time.time() < deadline - 10 and len(current_edges) > 0:
            rem_u, rem_v = random.choice(list(current_edges))
            adj[rem_u, rem_v] = False
            adj[rem_v, rem_u] = False
            current_edges.remove((rem_u, rem_v))
            
            non_edges_2 = [e for e in all_pairs if e not in current_edges]
            random.shuffle(non_edges_2)
            added = 0
            for u, v in non_edges_2:
                if added >= 2 or time.time() > deadline - 5:
                    break
                adj[u, v] = True
                adj[v, u] = True
                current_edges.add((u, v))
                if has_q3(adj):
                    adj[u, v] = False
                    adj[v, u] = False
                    current_edges.remove((u, v))
                else:
                    added += 1
                    if len(current_edges) > best_count:
                        best_count = len(current_edges)
                        best_edges = set(current_edges)
        
        random.shuffle(non_edges)
    
    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")
