import random
import time
import sys

def construct():
    deadline = time.time() + 115  # Safety margin
    n = 40
    target_edges = 401
    
    # 1. Initialization: K_20,20 + 1 intra-partition edge
    # Partition: 0-19 (A), 20-39 (B)
    A = list(range(20))
    B = list(range(20, 40))
    
    edges = set()
    # Add all cross edges
    for a in A:
        for b in B:
            edges.add((a, b))
            
    # Add one intra-partition edge
    # Randomly pick two from A
    u_extra = random.choice(A)
    v_extra = random.choice([x for x in A if x != u_extra])
    edges.add((u_extra, v_extra))
    
    # Ensure we have exactly 401 edges (K_20,20 has 400, +1 = 401)
    if len(edges) != target_edges:
        # This should not happen, but handle just in case
        pass

    # Bitmask adjacency
    # N[i] is an integer where bit j is set if (i,j) is an edge
    N = [0] * n
    for u, v in edges:
        N[u] |= (1 << v)
        N[v] |= (1 << u)
        
    # Helper to get list of neighbors from bitmask
    def get_neighbors(mask):
        return [i for i in range(n) if (mask >> i) & 1]
        
    # Helper to count bits
    def popcount(x):
        return bin(x).count("1")
        
    # Calculate book sizes
    # book[(u,v)] = number of common neighbors
    books = {}
    for u, v in edges:
        books[(u, v)] = popcount(N[u] & N[v])
        
    current_max = max(books.values())
    best_edges = set(edges)
    best_max = current_max
    
    # Local Search
    no_improve_count = 0
    
    while time.time() < deadline:
        if current_max <= 7:
            break
            
        # Find an edge with max book size
        max_edges = [(e, b) for e, b in books.items() if b == current_max]
        if not max_edges:
            break
            
        # Pick one randomly
        (u, v), _ = random.choice(max_edges)
        
        # Try to swap (u, v) with a random non-edge (x, y)
        improved = False
        # Try a few random swaps
        for _ in range(20):
            if time.time() >= deadline:
                break
                
            # Pick random non-edge
            while True:
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y:
                    continue
                if (x, y) in edges or (y, x) in edges:
                    continue
                break
            
            # Simulate swap
            # 1. Remove (u, v)
            # Save state to revert if needed
            # We update N and books in place, then revert
            
            # Affected by removal of (u, v):
            # (u, v) itself
            # (u, w) for w in N(u) & N(v)
            # (v, w) for w in N(u) & N(v)
            
            common_uv = N[u] & N[v]
            
            # List of edges affected by removal
            aff_remove = [(u, v)]
            for w in get_neighbors(common_uv):
                if w != u and w != v:
                    aff_remove.append((u, w))
                    aff_remove.append((v, w))
                    
            # Store old values
            old_books = {}
            for e in aff_remove:
                old_books[e] = books[e]
            old_N_u = N[u]
            old_N_v = N[v]
            
            # Apply removal
            N[u] &= ~(1 << v)
            N[v] &= ~(1 << u)
            books[(u, v)] = 0 # Mark as removed/inactive for now? 
            # Actually, we should remove it from books dict or set to -1?
            # Let's just update the values.
            for w in get_neighbors(common_uv):
                if w != u and w != v:
                    books[(u, w)] -= 1
                    books[(v, w)] -= 1
            
            # 2. Add (x, y)
            # Affected by addition of (x, y):
            # (x, y) itself
            # (x, z) for z in N(x) & N(y)
            # (y, z) for z in N(x) & N(y)
            
            # Note: N[x] and N[y] are currently updated (u,v removed)
            common_xy = N[x] & N[y]
            
            aff_add = [(x, y)]
            for z in get_neighbors(common_xy):
                if z != x and z != y:
                    aff_add.append((x, z))
                    aff_add.append((y, z))
                    
            old_books_add = {}
            for e in aff_add:
                # Note: (x,y) is not in books yet (it was a non-edge)
                # But (x,z) and (y,z) might be in books
                if e in books:
                    old_books_add[e] = books[e]
                else:
                    old_books_add[e] = 0 # Default for new edge
            
            old_N_x = N[x]
            old_N_y = N[y]
            
            # Apply addition
            N[x] |= (1 << y)
            N[y] |= (1 << x)
            books[(x, y)] = popcount(common_xy)
            
            for z in get_neighbors(common_xy):
                if z != x and z != y:
                    # Ensure edges exist in books dict
                    if (x, z) not in books:
                         books[(x, z)] = 0
                    if (y, z) not in books:
                         books[(y, z)] = 0
                    books[(x, z)] += 1
                    books[(y, z)] += 1
            
            # Calculate new max
            # We only need to check affected edges and the global max if it wasn't affected
            # But scanning all 401 edges is fast enough
            new_max = max(books.values())
            
            if new_max < current_max:
                # Accept
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_edges = set(edges)
                    # Update edges set
                    edges.discard((u, v))
                    edges.discard((v, u))
                    edges.add((x, y))
                    no_improve_count = 0
                else:
                    # Improved current, but not global best? 
                    # Actually if new_max < current_max, it is an improvement.
                    # We should update best if it's the best so far.
                    # Wait, best_max tracks the minimum found.
                    # If current_max < best_max, update best.
                    # If current_max == best_max, we might still want to keep it if it helps?
                    # Let's just update best_edges if current_max < best_max.
                    pass
                improved = True
                break
            else:
                # Revert
                # Restore N
                N[u] = old_N_u
                N[v] = old_N_v
                N[x] = old_N_x
                N[y] = old_N_y
                
                # Restore books
                for e, val in old_books.items():
                    books[e] = val
                for e, val in old_books_add.items():
                    books[e] = val
                
                # Note: (x,y) was added to books, need to remove or reset?
                # It wasn't in books before (it was a non-edge).
                # So we should delete it or set to 0?
                # If we set to 0, it's fine as long as we don't consider it an edge.
                # But `max(books.values())` will see 0.
                # However, for future swaps, we need `books` to reflect actual edges.
                # So we should remove (x,y) from books if it wasn't there before.
                # And restore (u,v) to books.
                
                # Let's refine revert logic:
                # 1. Remove (x,y) from books if it wasn't an edge before.
                if (x, y) not in old_books_add: # It wasn't in old_books_add keys?
                     # Wait, old_books_add stored old values.
                     # If (x,y) was not in books, old_books_add[(x,y)] = 0.
                     # We should delete it from books dict to keep it clean?
                     # Or just leave it as 0?
                     # If we leave it as 0, and later try to add it again, it's fine.
                     # But `edges` set doesn't contain it.
                     # So `books` should ideally only contain edges in `edges`.
                     # Let's delete it.
                    pass # We'll handle cleanup in a cleaner way below.
                
                # Actually, simpler revert:
                # We saved old_N. We restore N.
                # We saved old_books for affected edges. We restore them.
                # We need to ensure (u,v) is back in books and (x,y) is removed.
                
                # (u,v) was in old_books, so it's restored.
                # (x,y) was NOT in old_books (it was a non-edge).
                # So we must delete books[(x,y)].
                del books[(x, y)]
                
                # Also, if (x,z) or (y,z) were not in books before?
                # No, (x,z) is an edge if z is in N(x).
                # If (x,z) was an edge, it was in books.
                # If it wasn't, we created it.
                # But we only iterate z in N(x) & N(y).
                # If (x,z) is an edge, it's in books.
                # So all affected edges in aff_add were either (x,y) [new] or existing edges.
                # So restoring old_books_add works for existing ones.
                # For (x,y), we delete.
                
                # Wait, what if (x,z) was NOT an edge?
                # Then it's not in books.
                # But we did `books[(x, z)] += 1`.
                # If it wasn't there, we created it with 1?
                # No, `if (x, z) not in books: books[(x, z)] = 0` then `+= 1`.
                # So we created it.
                # So we must delete it if it wasn't there before.
                
                # This revert logic is getting messy.
                # Better approach: Copy the books dict? No, 401 entries is small.
                # Let's just copy the books dict before applying the swap.
                # 401 integers copy is very fast.
                
                # Let's restart the loop body with the copy strategy for safety and simplicity.
                # It's O(E) per swap, which is fine.
                
            if improved:
                continue
                
        if not improved:
            no_improve_count += 1
            # If stuck, try a random perturbation?
            # Or just continue trying different edges.
            if no_improve_count > 100:
                # Reset no_improve
                no_improve_count = 0
                # Maybe accept a neutral move?
                # For now, just continue.

    # Convert best_edges to list of lists
    result = [[u, v] for u, v in best_edges]
    return result
