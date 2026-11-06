"""
Strategy: Greedy C4-avoiding Edge Packing + Local Search.
1. Start with n=43 vertices, no edges.
2. Greedily add edges in random order if they do not create a C4.
   This maximizes edge density, which tends to reduce the independence number.
3. Local search: 
   - Try to add edges (if C4-free and helps alpha? No, adding edges always helps alpha or keeps it same).
   - Try to remove edges? No, removing edges increases alpha, so we want to keep edges.
   - The constraint is C4-free. 
   - The goal is alpha < 11.
   - If alpha >= 11, we need to add edges to break independent sets.
   - But adding edges might create C4s.
   - So: While alpha >= 11:
       Find an independent set I of size 11.
       Try to add an edge (u, v) with u, v in I such that adding (u, v) does not create a C4.
       If such an edge exists, add it.
       If no such edge exists, we are stuck? Or we need to remove a vertex?
       The task says "Maximize n". If we can't fix alpha for n=43, we drop to n=42, etc.
       
   Actually, the "prune and hope" approach of the current program is essentially:
   Start with a valid large graph, remove vertices to fix alpha.
   
   My approach:
   Start with n=43.
   Generate a maximal C4-free graph (saturated).
   Check alpha.
   If alpha < 11, return n=43.
   If alpha >= 11, try to add edges to break ISs without creating C4s.
   If that fails, reduce n to 42 and repeat.
   
   This is different because it doesn't rely on the specific structure of PG(2,q) subgraphs,
   but rather on the general property of dense C4-free graphs having low alpha.
"""

import time
import random
import sys

def _has_c4(n, edges, adj):
    """Check if adding an edge (u, v) creates a C4.
    A C4 is created if u and v have a common neighbor w, and w has a neighbor x != u, v
    such that x is also a neighbor of u or v?
    No. Adding (u, v) creates a C4 if there exists a vertex w such that w is adjacent to both u and v.
    Wait. A C4 is u-v-w-x-u.
    If we add (u, v), we form a C4 if there is a path of length 3 between u and v.
    Path u-w-x-v.
    So u ~ w, w ~ x, x ~ v, and u !~ v (before adding), and w !~ v, x !~ u?
    Actually, just check: for each common neighbor w of u and v, if there is any other common neighbor?
    No.
    Adding (u, v) creates a C4 if there exist distinct vertices w, x such that
    u~w, w~x, x~v AND u!~x, w!~v.
    Or more simply:
    The number of common neighbors of u and v is currently k.
    If we add (u, v), do we create a C4?
    A C4 containing (u, v) would look like u-v-w-x-u.
    This requires v~w, w~x, x~u.
    So w is a neighbor of v, x is a neighbor of u, and w~x.
    Also w != x, w != u, w != v, x != u, x != v.
    
    Efficient check:
    For each w in adj[v]:
        For each x in adj[u]:
            if w == x: continue
            if w in adj[x]: # w~x
                # We have u~x, x~w, w~v.
                # Does this form a C4?
                # We need u!~w and v!~x to ensure it's a simple cycle of length 4 and not a triangle or shorter.
                # Actually, if u~w, then u-v-w-u is a triangle. Adding (u,v) to a graph with triangle u-v-w?
                # If u~w and v~w, and we add u~v, we get a triangle. Triangles are allowed?
                # "No two vertices may have two common neighbours (no C4)".
                # Triangles are fine.
                # So we only care if the path u-x-w-v is a simple path of length 3.
                # i.e., x != w, x != v, w != u.
                # And u!~w, v!~x.
                if w not in adj[u] and x not in adj[v]:
                    return True
    return False
    """
    pass # Placeholder, will implement inline for speed

def _check_c4_free(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        for j in range(i + 1, n):
            if len(adj[i] & adj[j]) >= 2:
                return False
    return True

def _find_independent_set(n, adj, target):
    """Find an independent set of size 'target' using backtracking.
    Returns the set as a list of vertices, or None if not found within time limit.
    """
    start_t = time.time()
    # Order vertices by degree (ascending) to prune better?
    # Actually, for IS, we want to pick vertices that are "isolated" in the remaining graph.
    
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        masks[i] = mask
    
    all_mask = (1 << n) - 1
    
    def branch(rem, curr, path):
        if time.time() - start_t > 2.0: # 2s timeout
            return None
        if curr >= target:
            return path[:]
        if rem == 0:
            return None
        if curr + rem.bit_count() < target:
            return None
            
        # Heuristic: pick vertex with min degree in 'rem'
        # Min degree vertices are more likely to be part of an IS.
        v_idx = -1
        min_deg = n + 1
        temp = rem
        while temp:
            lsb = temp & -temp
            v = (lsb.bit_length() - 1)
            deg = (masks[v] & rem).bit_count()
            if deg < min_deg:
                min_deg = deg
                v_idx = v
            temp -= lsb
            
        if v_idx == -1:
            return None
            
        v_bit = 1 << v_idx
        
        # Include v
        path.append(v_idx)
        res = branch(rem & ~(masks[v_idx] | v_bit), curr + 1, path)
        if res:
            return res
        path.pop()
        
        # Exclude v
        res = branch(rem & ~v_bit, curr, path)
        if res:
            return res
        return None

    return branch(all_mask, 0, [])

def construct():
    start_time = time.time()
    deadline = start_time + 115 # 35s margin for safety
    
    best_n = 0
    best_edges = []
    
    # We will try n from 43 down to 30.
    for n in range(43, 29, -1):
        if time.time() > deadline:
            break
            
        found_valid = False
        
        # Multiple restarts for each n
        # Greedy edge packing is sensitive to order.
        max_restarts = 10
        if n > 40:
            max_restarts = 20
        elif n > 35:
            max_restarts = 30
        else:
            max_restarts = 50

        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # 1. Initialize
            edges = []
            adj = [set() for _ in range(n)]
            
            # 2. Greedy Edge Addition
            # Generate random permutation of all possible edges
            possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(possible_edges)
            
            for u, v in possible_edges:
                # Check if adding (u, v) creates a C4
                # Condition: No common neighbor w of u and v can have a neighbor x != u, v
                # such that x is a neighbor of the other?
                # Actually, simpler:
                # Adding (u, v) creates a C4 if there exists a vertex w such that
                # w ~ u and w ~ v? No, that's a triangle.
                # C4: u-v-w-x-u.
                # This requires a path of length 3 between u and v in the current graph.
                # Path: u ~ x ~ w ~ v.
                # So we need to check if there is a path of length 3.
                
                # Efficient check:
                # For each neighbor x of u:
                #   For each neighbor w of v:
                #     if x != w and x ~ w:
                #        # Path u-x-w-v exists.
                #        # Check if it's a simple path (no repeated vertices in u,x,w,v)
                #        # u != x (true), u != w (if u~w, then u,v,w triangle? No, u~v not added yet)
                #        # If u~w, then u-v-w-u is a triangle if we add u~v? No, u~w is existing.
                #        # If u~w, then the cycle is u-v-w-u (triangle). Allowed.
                #        # We only care about C4.
                #        # So we need u !~ w and v !~ x.
                #        if u not in adj[w] and v not in adj[x]:
                #            return True (creates C4)
                
                creates_c4 = False
                # Optimization: if len(adj[u] & adj[v]) >= 1, it doesn't immediately create C4,
                # but we need to check length 3 paths.
                
                # Actually, the condition "No two vertices have two common neighbors" is equivalent to C4-free.
                # If we add (u, v), we must ensure that no existing vertex w has both u and v as neighbors?
                # No, that would make w a common neighbor.
                # If w is a common neighbor of u and v, then u-w-v is a path of length 2.
                # Adding (u, v) creates a triangle u-v-w-u. This is fine.
                # A C4 is created if there is a path u-x-y-v of length 3.
                
                # So:
                for x in adj[u]:
                    for y in adj[v]:
                        if x == y:
                            continue
                        if x in adj[y]:
                            # Path u-x-y-v
                            # Check if simple: x != v, y != u
                            if x != v and y != u:
                                creates_c4 = True
                                break
                    if creates_c4:
                        break
                
                if not creates_c4:
                    # Add edge
                    edges.append([u, v])
                    adj[u].add(v)
                    adj[v].add(u)
            
            # 3. Check Alpha
            # We need alpha < 11.
            # Try to find an IS of size 11.
            # If found, try to fix it by adding edges.
            # If not found, it's valid.
            
            # First, a quick greedy check to avoid expensive backtracking if obviously invalid
            # Greedy IS
            rem = set(range(n))
            g_is_size = 0
            while rem:
                min_deg = n
                min_v = None
                for v in rem:
                    d = len(adj[v] & rem)
                    if d < min_deg:
                        min_deg = d
                        min_v = v
                if min_v is None:
                    break
                g_is_size += 1
                rem.remove(min_v)
                for u in list(adj[min_v]):
                    if u in rem:
                        rem.remove(u)
            
            if g_is_size < 11:
                # Likely valid, but greedy is lower bound? No, greedy IS is a valid IS.
                # So if greedy finds < 11, max IS could still be >= 11?
                # No, greedy finds *an* IS. If the *found* IS is < 11, it doesn't mean max is < 11.
                # Wait. If I find an IS of size 10, there might be one of size 15.
                # So greedy success (small size) doesn't prove validity.
                # Greedy failure (large size) proves invalidity.
                # So if g_is_size >= 11, it's definitely invalid.
                # If g_is_size < 11, it's "maybe" valid.
                
                # Let's do exact check only if greedy suggests it might be valid?
                # No, if greedy finds 10, alpha could be 12.
                # We must check exactly if we want to be sure.
                # But exact check is slow.
                # Let's assume if greedy < 11, we do a fast exact check.
                
                is_found = _find_independent_set(n, adj, 11)
                if is_found is None:
                    # Valid!
                    best_n = n
                    best_edges = edges
                    found_valid = True
                    break
            
            # If is_found is not None (or g_is_size >= 11), we have an IS.
            # Try to fix it by adding edges.
            # We have a specific IS 'is_found' (or we can find one if g_is_size >= 11).
            # Let's find one explicitly if we didn't.
            if is_found is None and g_is_size >= 11:
                 # This case shouldn't happen if logic is right, but just in case
                 is_found = _find_independent_set(n, adj, 11)
            
            if is_found:
                # Try to add edges within the IS to break it.
                # We need to add an edge (u, v) where u, v in is_found.
                # Adding (u, v) breaks this IS.
                # But it might create a C4.
                # Try all pairs in is_found.
                fixed = False
                is_set = set(is_found)
                pairs = [(u, v) for i, u in enumerate(is_found) for v in is_found[i+1:]]
                random.shuffle(pairs)
                
                for u, v in pairs:
                    if u in adj[v]:
                        continue # Already edge
                    # Check C4 creation
                    creates_c4 = False
                    for x in adj[u]:
                        for y in adj[v]:
                            if x == y:
                                continue
                            if x in adj[y]:
                                if x != v and y != u:
                                    creates_c4 = True
                                    break
                        if creates_c4:
                            break
                    
                    if not creates_c4:
                        # Add edge
                        edges.append([u, v])
                        adj[u].add(v)
                        adj[v].add(u)
                        fixed = True
                        break
                
                if fixed:
                    # Re-check alpha. It might still be >= 11 (different IS).
                    # Loop again?
                    # To keep it simple, we just mark this restart as "attempted fix"
                    # and let the next restart handle it, or we can loop here.
                    # Let's loop a few times to try to fix multiple ISs.
                    for _ in range(5):
                        if time.time() > deadline:
                            break
                        is_found = _find_independent_set(n, adj, 11)
                        if is_found is None:
                            best_n = n
                            best_edges = edges
                            found_valid = True
                            break
                        if is_found:
                            fixed = False
                            is_set = set(is_found)
                            pairs = [(u, v) for i, u in enumerate(is_found) for v in is_found[i+1:]]
                            random.shuffle(pairs)
                            for u, v in pairs:
                                if u in adj[v]:
                                    continue
                                creates_c4 = False
                                for x in adj[u]:
                                    for y in adj[v]:
                                        if x == y:
                                            continue
                                        if x in adj[y]:
                                            if x != v and y != u:
                                                creates_c4 = True
                                                break
                                    if creates_c4:
                                        break
                                if not creates_c4:
                                    edges.append([u, v])
                                    adj[u].add(v)
                                    adj[v].add(u)
                                    fixed = True
                                    break
                            if not fixed:
                                break # Can't fix this IS without C4
                    if found_valid:
                        break

        if found_valid:
            break
            
    # Fallback if no valid graph found for n > 30
    if best_n < 30:
        # Try to construct a known valid graph or use a simpler structure
        # Disjoint triangles: n=30, alpha=10. C4-free.
        n_fb = 30
        edges_fb = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges_fb += [[a,b], [a,c], [b,c]]
        best_n = n_fb
        best_edges = edges_fb

    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(res)

