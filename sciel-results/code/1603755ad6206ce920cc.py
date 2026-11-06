import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    # colors: -1 unassigned, 0/1 assigned
    colors = [-1] * n
    
    # Precompute adjacency: for each vertex, list of edges containing it
    # And for each edge, list of vertices
    edge_verts = [list(e) for e in edges]
    
    # Count degree of each vertex
    deg = [0] * n
    for e in edge_verts:
        for v in e:
            deg[v] += 1
            
    # Order vertices by degree descending to fail fast
    # But we need to assign in a specific order for backtracking?
    # Standard DFS assigns 0, 1, 2...
    # We can permute the vertices. Let's keep original indices but check constraints smartly.
    
    def check_conflict():
        for e in edge_verts:
            cnt0 = 0
            cnt1 = 0
            unassigned = 0
            for v in e:
                if colors[v] == 0: cnt0 += 1
                elif colors[v] == 1: cnt1 += 1
                else: unassigned += 1
            
            # If mono, conflict
            if cnt0 == 5 or cnt1 == 5:
                return True
            # If 4 of one color, the last must be the other
            # This is not a conflict yet, but a constraint.
            # We handle propagation separately or just let DFS handle it.
            # For simplicity and speed in Python, simple check is often enough if n is small.
            # However, propagation is key for speed.
            
        return False

    # Let's implement a solver with propagation
    # State: colors array
    # We will use a stack or recursion.
    
    def propagate():
        # Returns True if no conflict, False if conflict
        # Applies unit propagation: if 4 are 0, 5th must be 1
        # This might need multiple passes if one assignment triggers another.
        changed = True
        while changed:
            changed = False
            for e in edge_verts:
                cnt0 = 0
                cnt1 = 0
                unassigned_v = -1
                unassigned_count = 0
                for v in e:
                    if colors[v] == 0: cnt0 += 1
                    elif colors[v] == 1: cnt1 += 1
                    else:
                        unassigned_count += 1
                        if unassigned_count == 1:
                            unassigned_v = v
                        else:
                            unassigned_v = -1 # more than 1 unassigned
                
                if cnt0 == 4:
                    if unassigned_count == 1:
                        if colors[unassigned_v] == 0:
                            return False
                        if colors[unassigned_v] == -1:
                            colors[unassigned_v] = 1
                            changed = True
                    elif unassigned_count == 0:
                        # All assigned, check if mono
                        if cnt1 == 0: return False
                elif cnt1 == 4:
                    if unassigned_count == 1:
                        if colors[unassigned_v] == 1:
                            return False
                        if colors[unassigned_v] == -1:
                            colors[unassigned_v] = 0
                            changed = True
                    elif unassigned_count == 0:
                        if cnt0 == 0: return False
                elif cnt0 == 5 or cnt1 == 5:
                    return False
        return True

    def dfs(idx):
        if idx == n:
            return True
        
        # Try 0
        colors[idx] = 0
        if propagate():
            if dfs(idx + 1):
                return True
        colors[idx] = -1
        
        # Try 1
        colors[idx] = 1
        if propagate():
            if dfs(idx + 1):
                return True
        colors[idx] = -1
        
        return False

    # Optimization: Order vertices by degree
    # We need to map original indices to ordered indices
    # But the edges are fixed.
    # Let's just use the standard order 0..n-1.
    # For n=15, 2^15 = 32768, which is small. Propagation prunes heavily.
    
    return dfs(0)

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    best_m = 200
    best_n = 0
    best_edges = []
    
    # We will try to find a small non-2-colorable hypergraph.
    # Strategy:
    # 1. For n in 10..18:
    #    a. Generate random hypergraphs with m edges.
    #    b. If non-2-colorable, minimize it.
    #    c. Update best.
    
    # To ensure we find something, we start with a known construction or dense random.
    # The prompt says "best known construction has 51 edges".
    # We should aim for < 51.
    
    # Let's try to generate random hypergraphs and minimize them.
    # Minimization:
    # Given H (non-2-colorable), for each edge e in H:
    #   if H \ {e} is non-2-colorable, remove e.
    # This gives a minimal hypergraph.
    
    # To get a *minimum* (fewest edges), we try many random instances and minimize them.
    
    for n in range(10, 19):
        if time.time() > deadline:
            break
            
        # Number of trials
        trials = 50
        
        for t in range(trials):
            if time.time() > deadline:
                break
                
            # Choose m. Start with a value likely to be non-2-colorable.
            # For n=15, random hypergraph with m=40 is likely non-2-colorable?
            # Let's try m from 30 to 60.
            # Actually, let's just pick a random m in [30, 60] or similar.
            # Or better, start with a dense one and minimize.
            
            # Let's try to construct a random hypergraph with m edges.
            # If we start too sparse, it might be 2-colorable, and minimization doesn't apply (we can't minimize a 2-colorable graph to make it non-2-colorable by removing edges).
            # So we MUST start with a non-2-colorable hypergraph.
            
            # How to find a non-2-colorable hypergraph quickly?
            # Random hypergraphs become non-2-colorable when m is large enough.
            # For n=15, threshold is around m ~ n * 2^(n-5) / 5? No, that's for property B.
            # Actually, random hypergraphs are 2-colorable for small m.
            # Let's try m = 50.
            
            m_start = 50
            all_possible = list(combinations(range(n), 5))
            random.shuffle(all_possible)
            edges = [list(e) for e in all_possible[:m_start]]
            
            # Check if non-2-colorable
            if not is_2_colorable(n, edges):
                # It is non-2-colorable. Now minimize.
                current_edges = edges
                current_m = m_start
                
                # Iterative removal
                # We can try to remove edges in random order or fixed order.
                # Fixed order is deterministic. Random might find different cores.
                # Let's do multiple passes of random removal.
                
                for _ in range(10): # 10 passes of minimization
                    if time.time() > deadline:
                        break
                    # Shuffle edges to try different removal orders?
                    # Or just iterate through current list.
                    # If we remove an edge, the list shrinks.
                    # Let's create a set for faster removal?
                    # List is fine for m < 100.
                    
                    # We need to check if removing edge i keeps it non-2-colorable.
                    # This is expensive: O(m) checks, each O(2^n).
                    # Total O(m^2 * 2^n).
                    # For m=50, n=15, 2500 * 32768 ~ 80M ops. Might be slow in Python.
                    
                    # Optimization: Only try to remove if we have time.
                    # Also, we can stop if we reach a low m.
                    
                    # Let's just do one pass of removal.
                    # We iterate through edges.
                    # If we remove one, we restart the pass? Or continue?
                    # Continue is faster but might miss others.
                    # Let's do a single pass that checks each edge once.
                    
                    new_edges = []
                    for e in current_edges:
                        # Check if H \ {e} is non-2-colorable
                        # Construct H'
                        # To avoid copying list every time, we can use a "removed" flag?
                        # Or just filter.
                        
                        # Since m is small, filtering is fine.
                        # But calling is_2_colorable is the bottleneck.
                        
                        # Heuristic: If we already have a very small best_m, 
                        # and current_m is not much smaller, skip expensive checks?
                        # No, we want to minimize current.
                        
                        # Let's check.
                        # If time is running out, skip minimization and keep current?
                        # No, current might be 50, best is 126. 50 is better.
                        # But 30 is better than 50.
                        
                        # Let's limit the number of is_2_colorable calls.
                        if time.time() > deadline - 5:
                            break
                        
                        # Try removing e
                        # Create list without e
                        # We can just pass a generator or list comprehension
                        # But is_2_colorable expects a list.
                        
                        # Optimization:
                        # If we remove e, we check is_2_colorable(n, [x for x in current_edges if x != e])
                        # This is O(m) to create list + O(2^n) to check.
                        
                        # Let's just do it.
                        # If it's non-2-colorable, we remove e (don't add to new_edges).
                        # If it's 2-colorable, we keep e (add to new_edges).
                        
                        # Wait, if H is non-2-colorable, and H\{e} is 2-colorable, e is critical.
                        # If H\{e} is non-2-colorable, e is redundant.
                        
                        # So:
                        # If is_2_colorable(n, H\{e}) is False (i.e., non-2-colorable), then e is redundant. Remove it.
                        # If is_2_colorable(n, H\{e}) is True (i.e., 2-colorable), then e is critical. Keep it.
                        
                        # We want to KEEP e if it is critical.
                        # So if is_2_colorable(n, H\{e}) returns True, we keep e.
                        # If it returns False, we remove e.
                        
                        # Let's check time.
                        if time.time() > deadline - 10:
                            new_edges.append(e) # Keep to be safe? Or break?
                            continue
                            
                        h_minus_e = [x for x in current_edges if x != e]
                        if len(h_minus_e) == 0:
                            # Empty graph is 2-colorable
                            new_edges.append(e)
                        else:
                            if is_2_colorable(n, h_minus_e):
                                # 2-colorable, so e is needed
                                new_edges.append(e)
                            else:
                                # Non-2-colorable, e is redundant
                                pass # Remove e
                                
                    current_edges = new_edges
                    if len(current_edges) == current_m:
                        break # No edges removed in this pass
                    
                    current_m = len(current_edges)
                
                # Update best
                if current_m < best_m:
                    best_m = current_m
                    best_n = n
                    best_edges = current_edges
                    
                    # If we found a very good one, we can stop?
                    # Best known is 51. If we find < 51, great.
                    # If we find 29, that's the lower bound.
                    
    # Fallback if nothing found or best is still high
    if best_m > 126:
        # Use the seed's default (all edges for n=9) or a known small one?
        # The seed used n=9, all edges (126).
        # Let's just return the best we found.
        # If best_m is still 200 (initial), we need a valid one.
        # Let's construct a trivial non-2-colorable one: Complete hypergraph on 5 vertices?
        # K_5^5 has 1 edge. Is it non-2-colorable?
        # Edge {0,1,2,3,4}.
        # Coloring: 0=0, 1=0, 2=0, 3=0, 4=1. Edge is bichromatic.
        # So K_5^5 is 2-colorable.
        # We need a hypergraph where EVERY edge is mono in ANY coloring? No.
        # Non-2-colorable means: No coloring makes ALL edges bichromatic.
        # So there is no coloring where every edge has both 0 and 1.
        
        # Example: 2 edges {0,1,2,3,4} and {0,1,2,3,4}? No, distinct edges.
        # Consider n=5, edges: all 126?
        # Actually, the "Fano plane" analog?
        
        # Let's just use the seed's result if we didn't find a better one.
        # Seed: n=9, all 5-edges.
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_m = len(best_edges)

    return {'n': best_n, 'edges': best_edges}
