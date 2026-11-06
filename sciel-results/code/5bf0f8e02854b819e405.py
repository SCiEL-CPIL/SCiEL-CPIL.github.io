import random
import time

def construct():
    deadline = time.time() + 140  # Safety margin
    
    def try_solve(n):
        N = n * (n - 1) // 2
        # Initialize random colors
        colors = [random.randint(0, 4) for _ in range(N)]
        
        def get_idx(i, j):
            if i > j: i, j = j, i
            return i * n - i * (i + 1) // 2 + j - i - 1
            
        # Helper to check if a specific C4 exists in a color
        def is_c4(a, b, c, d, color):
            return (colors[get_idx(a, b)] == color and
                    colors[get_idx(b, c)] == color and
                    colors[get_idx(c, d)] == color and
                    colors[get_idx(d, a)] == color)

        # Count violations is expensive, so we just sample
        # We try to reduce violations by picking random vertices
        for _ in range(100000): # Max steps per n
            if time.time() > deadline:
                return None
            
            # Pick random 4 distinct vertices
            a, b, c, d = random.sample(range(n), 4)
            
            # Check the 3 possible C4s on these 4 vertices
            # C4 1: a-b-c-d-a
            for col in range(5):
                if is_c4(a, b, c, d, col):
                    # Found a violation. Pick a random edge to change
                    # Edges: ab, bc, cd, da
                    edge_idx = random.choice([get_idx(a, b), get_idx(b, c), get_idx(c, d), get_idx(d, a)])
                    old_col = colors[edge_idx]
                    new_col = random.choice([c for c in range(5) if c != old_col])
                    colors[edge_idx] = new_col
                    break
                if is_c4(a, b, d, c, col):
                    edge_idx = random.choice([get_idx(a, b), get_idx(b, d), get_idx(d, c), get_idx(c, a)])
                    old_col = colors[edge_idx]
                    new_col = random.choice([c for c in range(5) if c != old_col])
                    colors[edge_idx] = new_col
                    break
                if is_c4(a, c, b, d, col):
                    edge_idx = random.choice([get_idx(a, c), get_idx(c, b), get_idx(b, d), get_idx(d, a)])
                    old_col = colors[edge_idx]
                    new_col = random.choice([c for c in range(5) if c != old_col])
                    colors[edge_idx] = new_col
                    break
            
            # Periodically check if we are solved?
            # Checking all C4s is O(n^4) which is 26^4 ~ 456k.
            # We can do a full check every 1000 steps.
            if _ % 1000 == 0:
                solved = True
                for a in range(n):
                    for b in range(a+1, n):
                        for c in range(b+1, n):
                            for d in range(c+1, n):
                                # Check 3 C4s
                                for col in range(5):
                                    if is_c4(a, b, c, d, col) or is_c4(a, b, d, c, col) or is_c4(a, c, b, d, col):
                                        solved = False
                                        break
                                if not solved: break
                            if not solved: break
                        if not solved: break
                    if not solved: break
                if solved:
                    return colors
        
        return None

    # Try n=26
    result = try_solve(26)
    if result:
        return {'n': 26, 'colors': result}
        
    # If 26 fails, try 25
    if time.time() < deadline - 10:
        result = try_solve(25)
        if result:
            return {'n': 25, 'colors': result}

    # If all fails, return seed
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
