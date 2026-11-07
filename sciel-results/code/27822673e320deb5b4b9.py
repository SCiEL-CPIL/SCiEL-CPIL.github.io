import time
import random
from itertools import permutations

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    trees = instance_trees(20, 611)
    # trees[k-1] is the tree T_k
    # We want to embed T_20, T_19, ..., T_1
    # Result list index k-1 holds the embedding for T_k
    
    n = 20
    result = [None] * n
    used_edges = set()
    
    # Helper to check if an edge is available
    # Edges are undirected, so we store (min, max)
    def get_key(u, v):
        return (min(u, v), max(u, v))
    
    def is_edge_available(u, v):
        return get_key(u, v) not in used_edges
    
    # Backtracking function to find an embedding for a given tree
    # tree_edges: list of (u, v) in tree vertex space (0..k-1)
    # k: number of vertices in tree
    # available_vertices: list of vertices in K_n that can be used
    # We need to map tree vertices 0..k-1 to distinct graph vertices
    # such that all tree edges map to unused graph edges
    
    # To speed up, we sort tree vertices by degree (descending)
    # This is a heuristic to constrain the search space early
    
    def find_embedding(tree_edges, k, current_used_edges):
        if k == 1:
            # Any vertex works, but we must ensure it's not "used" in a way that conflicts?
            # T_1 has no edges, so any vertex is fine. 
            # However, the problem says "distinct int host vertices". 
            # If we pick a vertex, it's "occupied" by this tree, but edges are the only constraint for disjointness.
            # Wait, "edge-disjoint" means edges cannot be shared. Vertices CAN be shared.
            # So T_1 can map to any vertex, even if other trees use it.
            # But the return value is "list of k distinct int host vertices".
            # This implies the mapping is injective for that specific tree.
            # It does NOT imply the mapping is disjoint from other trees' vertices.
            # So for T_1, we can just pick vertex 0.
            return [0]
            
        if k == 2:
            # Tree is (0, 1). Need two distinct vertices u, v such that (u,v) is unused.
            for u in range(n):
                for v in range(u + 1, n):
                    if is_edge_available(u, v):
                        return [u, v]
            return None

        # Calculate degrees in the tree
        deg = [0] * k
        for u, v in tree_edges:
            deg[u] += 1
            deg[v] += 1
            
        # Order vertices by degree descending
        order = sorted(range(k), key=lambda x: -deg[x])
        
        # Mapping: tree_vertex -> graph_vertex
        mapping = {}
        
        # Heuristic: Try to map the highest degree vertex to a central graph vertex?
        # Or just iterate.
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == k:
                return True
                
            tv = order[idx]
            
            # Try all possible graph vertices for this tree vertex
            # We can prune: if tv is connected to already mapped vertices, check edges
            for gv in range(n):
                if gv in mapping.values():
                    continue
                
                # Check edges to already mapped neighbors
                valid = True
                for u, v in tree_edges:
                    if u == tv:
                        if v in mapping:
                            if not is_edge_available(gv, mapping[v]):
                                valid = False
                                break
                    elif v == tv:
                        if u in mapping:
                            if not is_edge_available(gv, mapping[u]):
                                valid = False
                                break
                if not valid:
                    continue
                
                mapping[tv] = gv
                if backtrack(idx + 1):
                    return True
                del mapping[tv]
                
            return False
        
        if backtrack(0):
            return [mapping[i] for i in range(k)]
        return None

    # Embed trees from largest to smallest
    # T_20 is a spanning tree. We can fix it to identity to save time and ensure it's embedded.
    # Identity: i -> i. Edges are (i, j) for edges in T_20.
    # Let's check if identity is valid for T_20. Yes, it uses the edges of T_20.
    # We will mark those edges as used.
    
    for k in range(n, 0, -1):
        if time.time() > deadline - 5:
            break
            
        tree_edges = trees[k - 1]
        
        if k == 20:
            # Try identity first
            emb = list(range(20))
            # Verify identity is valid (it always is for a tree on 20 vertices in K_20, 
            # unless we have prior used edges, but this is the first tree)
            # Mark edges as used
            for u, v in tree_edges:
                used_edges.add(get_key(u, v))
            result[k - 1] = emb
            continue
            
        # For k < 20, use backtracking
        emb = find_embedding(tree_edges, k, used_edges)
        
        if emb is not None:
            # Validate and mark edges
            for u, v in tree_edges:
                gu, gv = emb[u], emb[v]
                if not is_edge_available(gu, gv):
                    emb = None
                    break
                used_edges.add(get_key(gu, gv))
            
            if emb is not None:
                result[k - 1] = emb
            else:
                # Failed to embed, leave as None
                pass
        else:
            # Failed
            pass

    # Final check: ensure types are correct
    final_result = []
    for r in result:
        if r is None:
            final_result.append(None)
        else:
            # Ensure list of ints
            final_result.append([int(x) for x in r])
            
    return final_result

# Fallback if construct fails or times out internally? 
# The contract says return 20 entries.
# If we run out of time, we return what we have.

if __name__ == "__main__":
    res = construct()
    # Simple sanity check
    assert len(res) == 20
