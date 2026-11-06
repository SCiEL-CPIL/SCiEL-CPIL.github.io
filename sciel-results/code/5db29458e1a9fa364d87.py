import time
import random
import sys
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n = 16
    
    # Precompute Q3 bipartite pattern
    # Q3 vertices: 0..7 (3-bit)
    # Even popcount (left): 0(000), 3(011), 5(101), 6(110)
    # Odd popcount (right): 1(001), 2(010), 4(100), 7(111)
    # Q3 edges in bipartite form (left_idx, right_idx):
    # 0(000) connects to 1(001), 2(010), 4(100) -> (0,0), (0,1), (0,2)
    # 3(011) connects to 1(001), 2(010), 7(111) -> (1,0), (1,1), (1,3)
    # 5(101) connects to 1(001), 4(100), 7(111) -> (2,0), (2,2), (2,3)
    # 6(110) connects to 2(010), 4(100), 7(111) -> (3,1), (3,2), (3,3)
    
    # q3_pat[i][j] = 1 means Q3 left vertex i must connect to Q3 right vertex j
    q3_pat = [
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1]
    ]
    
    # Precompute all permutations for checking
    all_perms4 = list(permutations(range(4)))
    
    def check_q3_free(adj):
        """
        Check if graph with adjacency bitmasks adj[16] is Q3-free.
        Returns True if Q3-free.
        """
        # For each 8-vertex subset, check if it contains Q3
        # Optimization: only check subsets with enough edges
        for S in combinations(range(16), 8):
            S_list = list(S)
            S_set = frozenset(S)
            
            # Quick edge count: count edges within S
            edge_count = 0
            for u in S:
                for v in S:
                    if u < v and (adj[u] & (1 << v)):
                        edge_count += 1
            if edge_count < 12:
                continue
            
            # Try all 4+4 splits
            for left_idx in combinations(range(8), 4):
                left = [S_list[i] for i in left_idx]
                right = [S_list[i] for i in range(8) if i not in left_idx]
                
                # Quick check: each vertex must have degree >= 3 within the bipartite split
                ok = True
                for u in left:
                    cnt = 0
                    for v in right:
                        if adj[u] & (1 << v):
                            cnt += 1
                    if cnt < 3:
                        ok = False
                        break
                if not ok:
                    continue
                for v in right:
                    cnt = 0
                    for u in left:
                        if adj[u] & (1 << v):
                            cnt += 1
                    if cnt < 3:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Build 4x4 bipartite adjacency as bitmasks for speed
                bip_adj = []
                for i, u in enumerate(left):
                    row = 0
                    for j, v in enumerate(right):
                        if adj[u] & (1 << v):
                            row |= (1 << j)
                    bip_adj.append(row)
                
                # Check if Q3 pattern is a subgraph
                found = False
                for pi in all_perms4:
                    # Check if bip_adj rows permuted by pi can match
                    # For each column permutation sigma
                    for sigma in all_perms4:
                        ok2 = True
                        for i in range(4):
                            row = bip_adj[pi[i]]
                            for j in range(4):
                                if q3_pat[i][j]:
                                    if not (row & (1 << sigma[j])):
                                        ok2 = False
                                        break
                            if not ok2:
                                break
                        if ok2:
                            found = True
                            break
                    if found:
                        break
                if found:
                    return False
        
        return True
    
    # Construct Clebsch graph (Hamming distance 2 on 4-bit strings)
    def construct_clebsch():
        edges = set()
        for i in range(16):
            for j in range(i + 1, 16):
                xor = i ^ j
                bit_count = bin(xor).count('1')
                if bit_count == 2:
                    edges.add((i, j))
        return edges
    
    def build_adj(edge_set):
        adj = [0] * 16
        for u, v in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def try_add_edge(adj, edge_set, edge):
        """Try to add edge to edge_set, return new edge_set if Q3-free else None"""
        u, v = edge
        if (u, v) in edge_set:
            return None
        if (v, u) in edge_set:
            return None
        
        # Quick local check before full check
        # If adding (u,v) creates Q3, there must be a Q3 containing both u and v
        # For efficiency, just do the full check but with early termination
        
        new_edge_set = edge_set | {edge}
        new_adj = list(adj)  # copy
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        
        if check_q3_free(new_adj):
            return new_edge_set, new_adj
        return None
    
    # Start with Clebsch graph
    current_edges = construct_clebsch()
    current_adj = build_adj(current_edges)
    
    all_possible = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    
    # Randomized greedy edge addition with multiple restarts
    best_edges = current_edges
    best_count = len(current_edges)
    
    random.seed(42)
    
    # Strategy 1: Greedy addition from Clebsch
    for restart in range(3):
        if time.time() > deadline - 20:
            break
        
        edges = set(best_edges)
        adj = build_adj(edges)
        
        # Shuffle candidate edges
        candidates = [e for e in all_possible if e not in edges]
        random.shuffle(candidates)
        
        for edge in candidates:
            if time.time() > deadline - 15:
                break
            
            result = try_add_edge(adj, edges, edge)
            if result is not None:
                edges, adj = result
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = edges
    
    # Strategy 2: Local search - remove one edge, add one or two
    edges = set(best_edges)
    adj = build_adj(edges)
    
    improved = True
    max_passes = 20
    for _ in range(max_passes):
        if time.time() > deadline - 10:
            break
        if not improved:
            break
        improved = False
        
        # Try adding each missing edge
        for edge in all_possible:
            if time.time() > deadline - 10:
                break
            if edge in edges:
                continue
            
            result = try_add_edge(adj, edges, edge)
            if result is not None:
                edges, adj = result
                improved = True
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = edges
                break  # restart scan after improvement
    
    # Strategy 3: 1-remove-2-add local search
    if time.time() < deadline - 15:
        edges = set(best_edges)
        adj = build_adj(edges)
        
        improved = True
        max_swap_passes = 10
        for _ in range(max_swap_passes):
            if time.time() > deadline - 10:
                break
            if not improved:
                break
            improved = False
            
            # Try removing one edge and adding one or two
            for rem_edge in list(edges):
                if time.time() > deadline - 10:
                    break
                
                # Try adding one edge
                for add_edge in all_possible:
                    if add_edge in edges or add_edge == rem_edge:
                        continue
                    
                    new_edges = edges - {rem_edge} | {add_edge}
                    new_adj = build_adj(new_edges)
                    
                    if check_q3_free(new_adj):
                        if len(new_edges) > len(edges):
                            edges = new_edges
                            adj = new_adj
                            improved = True
                            if len(edges) > best_count:
                                best_count = len(edges)
                                best_edges = edges
                            break
                if improved:
                    break
            
            # Try removing one edge and adding two
            if not improved and time.time() < deadline - 12:
                for rem_edge in list(edges):
                    if time.time() > deadline - 12:
                        break
                    
                    missing = [e for e in all_possible if e not in edges and e != rem_edge]
                    # Sample a few pairs
                    if len(missing) > 50:
                        sampled = random.sample(missing, 50)
                    else:
                        sampled = missing
                    
                    for i, add1 in enumerate(sampled):
                        if time.time() > deadline - 12:
                            break
                        for add2 in sampled[i+1:]:
                            if time.time() > deadline - 12:
                                break
                            
                            new_edges = edges - {rem_edge} | {add1, add2}
                            new_adj = build_adj(new_edges)
                            
                            if check_q3_free(new_adj):
                                if len(new_edges) > len(edges):
                                    edges = new_edges
                                    adj = new_adj
                                    improved = True
                                    if len(edges) > best_count:
                                        best_count = len(edges)
                                        best_edges = edges
                                    break
                        if improved:
                            break
                    if improved:
                        break
    
    # Final: try to add more edges to best_edges
    edges = set(best_edges)
    adj = build_adj(edges)
    
    for edge in all_possible:
        if time.time() > deadline - 5:
            break
        if edge in edges:
            continue
        
        result = try_add_edge(adj, edges, edge)
        if result is not None:
            edges, adj = result
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
    
    return [list(e) for e in best_edges]
