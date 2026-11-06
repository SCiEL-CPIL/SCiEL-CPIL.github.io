import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    def color_index(n, i, j):
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def verify_colors(colors, n):
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    if colors[color_index(n, a, b)] == colors[color_index(n, a, c)] == colors[color_index(n, b, c)]:
                        return False
        return True

    # Try the 50-vertex construction based on Z_5 x Z_5
    n = 50
    colors = []
    
    # Map 50 vertices to (i, j) in Z_5 x Z_5
    # Vertex v corresponds to (v // 5, v % 5)
    
    for i in range(n):
        for j in range(i + 1, n):
            u_i, u_j = i // 5, i % 5
            v_i, v_j = j // 5, j % 5
            
            dx = (v_i - u_i) % 5
            dy = (v_j - u_j) % 5
            
            if dx == 0:
                c = 0
            elif dy == 0:
                c = 1
            else:
                # Calculate slope dy/dx mod 5
                # Inverse of dx mod 5
                inv_dx = pow(dx, -1, 5)
                slope = (dy * inv_dx) % 5
                if slope == 1:
                    c = 2
                elif slope == 2:
                    c = 3
                elif slope == 3:
                    c = 0
                elif slope == 4:
                    c = 1
                else:
                    c = 2 # Should not happen
            
            colors.append(c)

    if verify_colors(colors, n):
        return {'n': n, 'colors': colors}

    # Fallback to 32-vertex Greenwood-Gleason coloring
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map, x = {}, 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    n = 32
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if i // 16 == j // 16:
                colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)

    return {'n': n, 'colors': colors}
