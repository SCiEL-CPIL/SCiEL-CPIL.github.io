import time

N = 16

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return (N-1)*i - i*(i-1)//2 + (j-i-1)
    
    colors = [0] * (N*(N-1)//2)
    for i in range(N-1):
        for j in range(i+1, N):
            colors[edge_idx(i, j)] = i
    
    def check_4set(a, b, c, d):
        ab = colors[edge_idx(a,b)]
        ac = colors[edge_idx(a,c)]
        ad = colors[edge_idx(a,d)]
        bc = colors[edge_idx(b,c)]
        bd = colors[edge_idx(b,d)]
        cd = colors[edge_idx(c,d)]
        if len({ab, bc, cd, ad}) == 4: return True
        if len({ab, bd, cd, ac}) == 4: return True
        if len({ac, bc, bd, ad}) == 4: return True
        return False
    
    def can_set_edge(i, j, new_color):
        idx = edge_idx(i, j)
        old = colors[idx]
        colors[idx] = new_color
        others = [v for v in range(N) if v != i and v != j]
        for k_idx in range(len(others)):
            k = others[k_idx]
            for l_idx in range(k_idx+1, len(others)):
                l = others[l_idx]
                if check_4set(i, j, k, l):
                    colors[idx] = old
                    return False
        colors[idx] = old
        return True
    
    used_colors = set(range(15))
    edge_list = [(i, j) for i in range(N-1) for j in range(i+1, N)]
    
    for _ in range(20):
        if time.time() > deadline - 3.0:
            break
        improved = False
        for (i, j) in edge_list:
            if time.time() > deadline - 3.0:
                break
            idx = edge_idx(i, j)
            for new_c in range(15, 120):
                if new_c not in used_colors:
                    if can_set_edge(i, j, new_c):
                        colors[idx] = new_c
                        used_colors.add(new_c)
                        improved = True
                    break
        if not improved:
            break
    
    return colors
# EVOLVE-BLOCK-END
