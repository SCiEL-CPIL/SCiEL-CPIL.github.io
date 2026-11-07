# EVOLVE-BLOCK-START
import random
import itertools
import sys

def construct():
    n = 15
    all_edges = list(itertools.combinations(range(n), 2))
    edge_index = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    # Precompute bitmasks for vertices
    def bit(i):
        return 1 << i

    def is_valid(adj, degs, n):
        """
        adj: list of int bitmasks
        degs: list of int degrees
        Returns True if no forbidden 3-path exists.
        """
        # Precompute degree groups as bitmasks
        max_deg = max(degs) if degs else 0
        deg_groups = [0] * (max_deg + 1)
        for i in range(n):
            d = degs[i]
            deg_groups[d] |= bit(i)
            
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & bit(v)):
                    continue
                
                # Path s-u-v-t
                # s in adj[u] \ {v}
                # t in adj[v] \ {u, s}
                
                S_mask = adj[u] & ~bit(v)
                T_mask = adj[v] & ~bit(u)
                
                # Check for each degree d
                for d in range(max_deg + 1):
                    A = S_mask & deg_groups[d]
                    B = T_mask & deg_groups[d]
                    
                    if A and B:
                        # Violation if there exists s in A, t in B with s != t
                        # If A and B share a vertex, that vertex is s=t, which is allowed?
                        # No, path s-u-v-t requires s != t.
                        # If A == B and popcount(A) == 1, then the only candidate is s=t, which is not a simple path endpoint pair (s!=t).
                        # So if A == B and popcount(A) == 1, it's valid.
                        # Otherwise, there is a pair s != t.
                        if not (A == B and A.bit_count() == 1):
                            return False
        return True

    def get_adj_degs(edge_set, n):
        adj = [0] * n
        degs = [0] * n
        for u, v in edge_set:
            adj[u] |= bit(v)
            adj[v] |= bit(u)
            degs[u] += 1
            degs[v] += 1
        return adj, degs

    def greedy_add(current_edges, all_edges, n):
        current = set(current_edges)
        adj, degs = get_adj_degs(current, n)
        
        improved = True
        while improved:
            improved = False
            # Get missing edges
            missing = [e for e in all_edges if e not in current]
            # Heuristic: Try edges connecting vertices with lower degrees first?
            # Or just random? Random is robust.
            random.shuffle(missing)
            
            for e in missing:
                new_edges = current | {e}
                new_adj, new_degs = get_adj_degs(new_edges, n)
                if is_valid(new_adj, new_degs, n):
                    current = new_edges
                    adj, degs = new_adj, new_degs
                    improved = True
        return current

    def local_search(current_edges, all_edges, n, max_iters=500):
        current = set(current_edges)
        adj, degs = get_adj_degs(current, n)
        best = current
        best_size = len(current)
        
        for _ in range(max_iters):
            # Try to add
            added = False
            missing = [e for e in all_edges if e not in current]
            random.shuffle(missing)
            for e in missing:
                new_edges = current | {e}
                new_adj, new_degs = get_adj_degs(new_edges, n)
                if is_valid(new_adj, new_degs, n):
                    current = new_edges
                    adj, degs = new_adj, new_degs
                    added = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best = set(current)
                    break # Restart add loop if one added? Or continue? 
                    # Continuing is better to greedily fill.
            
            if added:
                continue
            
            # If no add possible, try a swap
            # Pick a random edge to remove
            if not current:
                break
                
            e_out = random.choice(list(current))
            # Pick a random edge to add
            missing = [e for e in all_edges if e not in current]
            if not missing:
                break
            e_in = random.choice(missing)
            
            # Try swap
            new_edges = (current - {e_out}) | {e_in}
            new_adj, new_degs = get_adj_degs(new_edges, n)
            if is_valid(new_adj, new_degs, n):
                # Accept swap
                current = new_edges
                adj, degs = new_adj, new_degs
                # Note: Size is same. We hope this allows future adds.
                
            # Occasionally, try removing an edge to see if we can add 2?
            # This is expensive. The swap is a good proxy.
            
            # If we haven't improved in a while, break?
            # The loop is bounded.
            
        return best

    random.seed(42)
    best_edges = set()
    best_count = 0
    
    # Multiple restarts
    for trial in range(300):
        if trial % 10 == 0:
            start_edges = set()
        else:
            if best_count > 0:
                start_edges = set(best_edges)
                # Perturb
                k = random.randint(2, 5)
                edges_list = list(start_edges)
                random.shuffle(edges_list)
                for i in range(min(k, len(edges_list))):
                    start_edges.remove(edges_list[i])
            else:
                start_edges = set()
                
        # 1. Greedy Add
        res1 = greedy_add(start_edges, all_edges, n)
        
        # 2. Local Search (Swaps)
        res2 = local_search(res1, all_edges, n, max_iters=100)
        
        # 3. Greedy Add again after swaps
        res3 = greedy_add(res2, all_edges, n)
        
        if len(res3) > best_count:
            best_count = len(res3)
            best_edges = res3

    result = [[u, v] for (u, v) in best_edges]
    final_result = []
    for edge in result:
        u = int(edge[0])
        v = int(edge[1])
        final_result.append([u, v])
        
    return final_result
# EVOLVE-BLOCK-END
