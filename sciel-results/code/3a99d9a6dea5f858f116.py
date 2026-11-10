import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Seed: (i + j) % 17
    # Pairs (i, j) for 0 <= i < j < 16
    # Lexicographic order: i=0, j=1..15; i=1, j=2..15; ...
    
    def generate_seed():
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    def is_valid(colours):
        # colours is a list of 120 ints
        # Check all K4s
        # Vertices 0..15
        # Edges are indexed by (i, j)
        # We need a fast way to get edge colour for (i, j)
        # Precompute index for (i, j)
        # idx(i, j) = i * 16 - i*(i+1)/2 + (j - i - 1)
        # Or just build a dict/list of lists
        
        # Let's build a 16x16 matrix of colours, 0 for non-edges
        # Actually, a list of lists is fine.
        c = [[0]*16 for _ in range(16)]
        idx = 0
        for i in range(16):
            for j in range(i+1, 16):
                c[i][j] = colours[idx]
                c[j][i] = colours[idx]
                idx += 1
        
        for combo in itertools.combinations(range(16), 4):
            a, b, c_v, d = combo
            e1 = c[a][b]
            e2 = c[a][c_v]
            e3 = c[a][d]
            e4 = c[b][c_v]
            e5 = c[b][d]
            e6 = c[c_v][d]
            if len({e1, e2, e3, e4, e5, e6}) < 5:
                return False
        return True

    current = generate_seed()
    
    # Greedy merging
    # We try to merge colours to reduce the count.
    # We try all pairs of existing colours.
    
    for _ in range(100): # Max iterations
        if time.time() > deadline:
            break
            
        current_colours = set(current)
        if len(current_colours) <= 13:
            break # Can't go below 13 theoretically? Actually 13 is lower bound.
            
        sorted_colours = sorted(current_colours)
        merged = False
        
        # Try all pairs
        for i in range(len(sorted_colours)):
            for j in range(i+1, len(sorted_colours)):
                if time.time() > deadline:
                    break
                c1 = sorted_colours[i]
                c2 = sorted_colours[j]
                
                # Map c1 -> c2
                new_colours = [c2 if x == c1 else x for x in current]
                
                if is_valid(new_colours):
                    current = new_colours
                    merged = True
                    break
            if merged:
                break
                
        if not merged:
            break
            
    return current
