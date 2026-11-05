import time
import sys

def construct():
    # GF(25) construction
    # Field is GF(5)[x] / (x^2 + 2)
    # alpha^2 = 3 (since -2 = 3 mod 5)
    
    def gf25_add(a, b):
        return ((a[0] + b[0]) % 5, (a[1] + b[1]) % 5)
    
    def gf25_sub(a, b):
        return ((a[0] - b[0]) % 5, (a[1] - b[1]) % 5)
    
    def gf25_mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha)
        # = a0*b0 + (a0*b1 + a1*b0)*alpha + a1*b1*alpha^2
        # alpha^2 = 3
        a0, a1 = a
        b0, b1 = b
        res0 = (a0 * b0 + 3 * a1 * b1) % 5
        res1 = (a0 * b1 + a1 * b0) % 5
        return (res0, res1)
        
    def gf25_norm(a):
        # N(a) = a * a^5 ? No, N(a) = a * conjugate(a)
        # conjugate(a+b*alpha) = a-b*alpha
        # N(a) = (a+b*alpha)(a-b*alpha) = a^2 - b^2*alpha^2 = a^2 - 3b^2 = a^2 + 2b^2
        a0, a1 = a
        return (a0 * a0 + 2 * a1 * a1) % 5

    # Generate all elements
    elements = []
    for a in range(5):
        for b in range(5):
            elements.append((a, b))
            
    zero = (0, 0)
    non_zero = [e for e in elements if e != zero]
    
    # Partition by norm
    norm_sets = {1: [], 2: [], 3: [], 4: []}
    for e in non_zero:
        n = gf25_norm(e)
        norm_sets[n].append(e)
        
    # We have 4 sets. We need 5 colors.
    # Split norm set 1 (size 6) into two sets of size 3.
    # The choice of split doesn't matter for C4-freeness as long as the original set is C4-free.
    set_1a = norm_sets[1][:3]
    set_1b = norm_sets[1][3:]
    
    # Assign colors
    # Color 0: set_1a
    # Color 1: set_1b
    # Color 2: norm_sets[2]
    # Color 3: norm_sets[3]
    # Color 4: norm_sets[4]
    
    color_map = {}
    for i, e in enumerate(set_1a):
        color_map[e] = 0
    for i, e in enumerate(set_1b):
        color_map[e] = 1
    for i, e in enumerate(norm_sets[2]):
        color_map[e] = 2
    for i, e in enumerate(norm_sets[3]):
        color_map[e] = 3
    for i, e in enumerate(norm_sets[4]):
        color_map[e] = 4
        
    n_vertices = 26
    colors = []
    
    # Vertices 0..24 are the field elements
    # Vertex 25 is infinity
    
    # Edges between field elements
    for i in range(25):
        for j in range(i + 1, 25):
            vi = elements[i]
            vj = elements[j]
            diff = gf25_sub(vj, vi)
            # diff is non-zero
            c = color_map[diff]
            colors.append(c)
            
    # Edges between field elements and infinity
    for i in range(25):
        vi = elements[i]
        # Color determined by vi (or -vi, same set)
        c = color_map[vi]
        colors.append(c)
        
    return {'n': n_vertices, 'colors': colors}

if __name__ == '__main__':
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end-start}")
    # Verify
    n = res['n']
    colors = res['colors']
    # Build adjacency
    adj = [[-1]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            adj[i][j] = colors[idx]
            adj[j][i] = colors[idx]
            idx += 1
            
    # Check for C4
    found_c4 = False
    for c in range(5):
        # For each pair of vertices, check common neighbors
        # If they have >= 2 common neighbors of color c, then C4 exists
        # This is O(n^3) which is 26^3 ~ 17576, very fast.
        for i in range(n):
            for j in range(i+1, n):
                common = 0
                for k in range(n):
                    if k != i and k != j:
                        if adj[i][k] == c and adj[j][k] == c:
                            common += 1
                            if common >= 2:
                                found_c4 = True
                                print(f"C4 found in color {c} with vertices {i}, {j} and common neighbor {k}")
                                break
                if common >= 2:
                    break
            if found_c4:
                break
        if found_c4:
            break
            
    if not found_c4:
        print("Valid coloring found!")
    else:
        print("Invalid coloring!")
