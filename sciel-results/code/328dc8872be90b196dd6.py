import time
import sys

def construct():
    """
    Constructs a 4-coloring of K_n without monochromatic triangles.
    Returns a dict with 'n' and 'colors'.
    """
    # Deadline: 110 seconds to be safe against the 120s kill.
    deadline = time.time() + 110
    
    # --- Step 1: Generate the base K32 coloring (Greenwood-Gleason / Clebsch) ---
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                base_colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                base_colors.append(3)
    
    n_current = 32
    colors = base_colors[:]
    
    # Precompute edge indices for K_n for n up to 62
    # index(i, j, n) for i < j
    # We will use a flat list `M` of size n*(n-1)//2
    # To speed up access, we can precompute the offset for each row i in a given n.
    
    def get_row_offset(i, n):
        # Number of edges before row i
        # Sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
        return i * (n - 1) - i * (i - 1) // 2

    def get_edge_index(i, j, n):
        # i < j
        return get_row_offset(i, n) + (j - i - 1)

    # Convert base_colors to a 2D-like structure for faster access?
    # Actually, flat list with precomputed offsets is fast enough if we cache offsets.
    
    def search(n_start, colors_list, time_limit):
        """
        Tries to extend the coloring from n_start to larger n using backtracking.
        Returns (max_n_found, best_colors)
        """
        max_n = n_start
        best_colors = colors_list[:]
        
        # We will use a recursive function. To avoid deep recursion overhead, 
        # we might need to be careful, but n <= 62 is fine for recursion depth.
        
        def backtrack(n, cols):
            nonlocal max_n, best_colors
            
            if time.time() > time_limit:
                return False
            
            if n > max_n:
                max_n = n
                best_colors = cols[:]
            
            # If we have reached 62, stop
            if n >= 62:
                return True
                
            # Try to add vertex n
            # Edges: (0, n), (1, n), ..., (n-1, n)
            # We assign colors for these edges sequentially.
            
            # Precompute indices for the new edges in the new list (size for K_{n+1})
            # The new list will have size (n+1)*n//2
            # The old list `cols` has size n*(n-1)//2
            # The new edges are appended at the end.
            
            # We maintain a partial assignment `new_edges` for the new vertex.
            
            def assign(idx, new_edges):
                if time.time() > time_limit:
                    return False
                
                if idx == n:
                    # Successfully assigned all edges for new vertex n
                    # Construct the new full color list
                    # The new edges are (0,n), (1,n), ..., (n-1,n)
                    # In the flat list for K_{n+1}, these are at the end.
                    # Indices: 
                    # (0, n) is index get_row_offset(0, n+1) + (n-0-1) = 0 + n-1 = n-1? 
                    # No, let's just append.
                    # The flat list order is (0,1), (0,2)...(0,n-1), (1,2)...(1,n-1), ..., (n-1, n)
                    # The new edges are (0,n), (1,n), ..., (n-1,n).
                    # In K_{n+1}, the edges are:
                    # Row 0: (0,1)...(0,n) -> last is (0,n)
                    # Row 1: (1,2)...(1,n) -> last is (1,n)
                    # ...
                    # Row n-1: (n-1,n) -> only (n-1,n)
                    # So the new edges are NOT contiguous at the end of the list.
                    # Wait, the flat list is lexicographic: (0,1), (0,2), ..., (0,n-1), (1,2), ..., (1,n-1), ..., (n-1, n-1)? No, (n-1, n).
                    # For K_n, the last edge is (n-2, n-1).
                    # For K_{n+1}, the new edges are (0,n), (1,n), ..., (n-1,n).
                    # Where do they fit?
                    # (0,n) is at index: offset(0, n+1) + (n-0-1) = n-1.
                    # (1,n) is at index: offset(1, n+1) + (n-1-1) = (n-1) - 0 + n-2 = 2n-3?
                    # It's easier to build the new list from scratch or insert.
                    # Building from scratch is O(n^2) which is small (62^2 ~ 3800).
                    
                    new_cols = []
                    # Reconstruct full list for K_{n+1}
                    # We have `cols` for K_n and `new_edges` for (i, n).
                    # We need to merge them.
                    # It's faster to just create a 2D matrix, fill it, and flatten.
                    # But we are in a hot loop.
                    
                    # Let's optimize: 
                    # The new list is `cols` with new edges inserted at specific positions.
                    # Or, simpler: just build the new list by iterating all pairs.
                    # Since n is small, this is fine.
                    
                    # Create a 2D array for K_{n+1}
                    # We can reuse the logic from before but it was slow.
                    # Let's do it in-place on a list?
                    
                    # Actually, we can just append the new edges to a temporary list and then 
                    # we need to interleave them? No, the flat list order is fixed.
                    # (0,1), (0,2)...(0,n-1), (0,n)
                    # (1,2)...(1,n-1), (1,n)
                    # ...
                    # (n-1, n)
                    
                    # So we can build `new_cols` by iterating i from 0 to n-1:
                    # For each i, append edges (i, i+1)...(i, n)
                    # The edges (i, i+1)...(i, n-1) are already in `cols`.
                    # The edge (i, n) is `new_edges[i]`.
                    
                    new_cols = []
                    for i in range(n):
                        # Append edges (i, i+1) to (i, n-1) from old cols
                        # How many are there? n-1-i
                        # They start at index get_row_offset(i, n)
                        # End at get_row_offset(i, n) + (n-1-i)
                        start_idx = get_row_offset(i, n)
                        end_idx = start_idx + (n - 1 - i)
                        new_cols.extend(cols[start_idx:end_idx])
                        # Append new edge (i, n)
                        new_cols.append(new_edges[i])
                    
                    return backtrack(n + 1, new_cols)
                
                # Determine allowed colors for edge (idx, n)
                # Constraint: For any j < idx, triangle (j, idx, n) must not be monochromatic.
                # color(idx, n) != color(idx, j) if color(j, n) == color(idx, j)
                
                allowed = [0, 1, 2, 3]
                # We need color(idx, j) from `cols` (K_n)
                # Index of (j, idx) in K_n: get_row_offset(j, n) + (idx - j - 1)
                
                for j in range(idx):
                    c_jn = new_edges[j]
                    c_ij = cols[get_row_offset(j, n) + (idx - j - 1)]
                    if c_ij == c_jn:
                        if c_ij in allowed:
                            allowed.remove(c_ij)
                
                for c in allowed:
                    if assign(idx + 1, new_edges + [c]):
                        return True
                return False
            
            if assign(0, []):
                return True
            return False

        backtrack(n_start, colors_list)
        return max_n, best_colors

    # Run the search
    # Start from 32
    found_n, found_colors = search(32, colors, deadline)
    
    # Safety check: if found_n is 32, ensure colors are valid K32
    if found_n == 32:
        found_colors = colors[:]

    return {'n': found_n, 'colors': found_colors}

if __name__ == '__main__':
    res = construct()
    print(f"Found n={res['n']}")
