import time
import random
from itertools import combinations

random.seed(42)
START_TIME = time.time()
TIME_LIMIT = 115

def build_adj(edges, n=16):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def is_triangle_free_bitmask(adj, n=16):
    for u in range(n):
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            common = adj[v] & adj[u]
            if common & ~(1 << u) & ~(1 << v):
                return False
    return True

def is_not_3_colorable_bitmask(adj, n=16):
    colors = [-1] * n
    
    def dfs(uncolored_mask):
        if uncolored_mask == 0:
            return False
        
        # Select vertex with most colored neighbors
        best_v = -1
        best_score = -1
        m = uncolored_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            score = 0
            nb_mask = adj[v]
            mm = nb_mask
            while mm:
                lsb_nb = mm & -mm
                nb = lsb_nb.bit_length() - 1
                mm ^= lsb_nb
                if colors[nb] != -1:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v
        
        v = best_v
        
        forbidden = 0
        nb_mask = adj[v]
        mm = nb_mask
        while mm:
            lsb_nb = mm & -mm
            nb = lsb_nb.bit_length() - 1
            mm ^= lsb_nb
            if colors[nb] != -1:
                forbidden |= 1 << colors[nb]
        
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if not dfs(uncolored_mask & ~(1 << v)):
                    return False
                colors[v] = -1
        
        return True
    
    initial_mask = (1 << n) - 1
    return dfs(initial_mask)

def greedy_3_colorable(adj, n=16, max_trials=50):
    for _ in range(max_trials):
        order = list(range(n))
        random.shuffle(order)
        colors = [-1] * n
        for node in order:
            used = 0
            nb_mask = adj[node]
            mm = nb_mask
            while mm:
                lsb = mm & -mm
                nb = lsb.bit_length() - 1
                mm ^= lsb
                if colors[nb] != -1:
                    used |= 1 << colors[nb]
            for c in range(3):
                if not (used & (1 << c)):
                    colors[node] = c
                    break
            else:
                break
        else:
            return True
    return False

def is_not_3_colorable_fast(adj, n=16):
    if greedy_3_colorable(adj, n, max_trials=20):
        return False
    return is_not_3_colorable_bitmask(adj, n)

def build_groetzsch_edges():
    edges = [[0,1],[1,2],[2,3],[3,4],[4,0]]
    for u, v in [[0,1],[1,2],[2,3],[3,4],[4,0]]:
        edges.append([5+u, v])
        edges.append([5+v, u])
    for u in range(5):
        edges.append([10, 5+u])
    return edges

def get_groetzsch_adj(n=11):
    edges = build_groetzsch_edges()
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def get_maximal_independent_sets(adj, n):
    mis = []
    # Simple backtracking to find maximal independent sets
    # This might be slow if n is large, but n=16 is small.
    # Actually, we only need the independent sets that are maximal.
    # For the purpose of adding a new vertex, we want large independent sets.
    
    # Let's just generate all independent sets and filter for maximal?
    # Or use a heuristic: try to add vertices to an empty set.
    
    # Given the small size, we can just iterate through all subsets? 2^16 = 65536.
    # For each subset, check if it's independent.
    # This is fast.
    
    all_is = []
    for mask in range(1 << n):
        is_independent = True
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            m ^= lsb
            if adj[u] & mask:
                is_independent = False
                break
        if is_independent:
            # Check if maximal
            is_maximal = True
            # Check if any vertex outside mask can be added
            # Actually, we just want the largest ones.
            all_is.append((mask.bit_count(), mask))
            
    # Sort by size
    all_is.sort(reverse=True)
    return all_is

def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = 115
    
    n = 16
    g_adj = get_groetzsch_adj(11)
    
    # Extend the Grötzsch graph with 5 more vertices
    # We will use a recursive search to add vertices 11, 12, 13, 14, 15
    # At each step, we try to connect the new vertex to a large independent set of the current graph
    
    best_edges = build_groetzsch_edges()
    best_count = len(best_edges)
    
    def search(adj, current_edges, next_v):
        nonlocal best_edges, best_count
        
        if time.time() > START_TIME + TIME_LIMIT:
            return
        
        if next_v == n:
            # Check if not 3-colorable
            # It should be, since it contains Grötzsch
            # But let's be safe
            if is_not_3_colorable_fast(adj, n):
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = list(current_edges)
            return
        
        # Get independent sets of the current graph (vertices 0..next_v-1)
        # We only care about vertices 0..next_v-1
        curr_n = next_v
        curr_adj = adj[:curr_n]
        
        # Find top independent sets
        is_list = get_maximal_independent_sets(curr_adj, curr_n)
        
        # Limit the number of candidates to explore to keep it fast
        # Try the top 20 largest independent sets
        for size, mask in is_list[:20]:
            if size == 0:
                break
                
            # Create new adj
            new_adj = adj[:]
            new_mask = 1 << next_v
            
            # Add edges from next_v to all vertices in mask
            m = mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                m ^= lsb
                new_adj[next_v] |= 1 << u
                new_adj[u] |= new_mask
                current_edges.append([next_v, u])
            
            # Recurse
            search(new_adj, current_edges, next_v + 1)
            
            # Backtrack
            m = mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                m ^= lsb
                current_edges.pop()
            
            # Check time
            if time.time() > START_TIME + TIME_LIMIT:
                return
        
        # Also try adding the vertex with NO edges to existing vertices
        # This allows it to connect to future vertices
        # But we want to maximize edges, so connecting to a large IS is better
        # However, maybe a smaller IS now allows a much larger IS later?
        # The search above explores different ISs, so it should cover this.
        
    search(g_adj, list(build_groetzsch_edges()), 11)
    
    # If the search didn't find anything better, return the best found
    # We should also try local search on the best found graph
    if best_edges:
        adj = build_adj(best_edges, n)
        current_edges = list(best_edges)
        
        # Simple local search: try to add any edge that doesn't create a triangle
        # and check if it remains not 3-colorable
        improved = True
        while improved and (time.time() < START_TIME + TIME_LIMIT):
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] & (1 << v)):
                        if (adj[u] & adj[v]) == 0:
                            # Try adding edge
                            adj[u] |= 1 << v
                            adj[v] |= 1 << u
                            current_edges.append([u, v])
                            
                            if is_not_3_colorable_fast(adj, n):
                                improved = True
                            else:
                                adj[u] ^= 1 << v
                                adj[v] ^= 1 << u
                                current_edges.pop()
                                
                    if time.time() > START_TIME + TIME_LIMIT:
                        return best_edges
        best_edges = current_edges
        
    return best_edges

# Run construct to ensure it works
if __name__ == "__main__":
    res = construct()
    print(len(res))
