import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    """Return the lexicographic index of edge (i,j) where 0 <= i < j < N."""
    if i > j:
        i, j = j, i
    # Row i has edges (i, i+1), (i, i+2), ..., (i, N-1)
    # Number of edges before row i: sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - i*(i-1)//2
    # Position within row i: j - i - 1
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def idx_to_edge(idx):
    """Convert lexicographic index back to (i, j) with i < j."""
    i = 0
    remaining = idx
    while True:
        row_len = N - 1 - i
        if remaining < row_len:
            j = i + 1 + remaining
            return (i, j)
        remaining -= row_len
        i += 1

def construct():
    """
    Construct a coloring of K_16 edges with no rainbow 4-cycle.
    Strategy:
      1. Build a Gallai-type recursive coloring (guaranteed valid, ~19 colors).
      2. Try modular arithmetic initializations with local search to exceed 19.
      3. Return the best valid coloring found.
    """
    deadline = time.time() + 75  # safety margin under 90s

    # --- Helper: check validity ---
    def _is_valid(colors):
        for quad in combinations(range(N), 4):
            a, b, c, d = quad
            c1 = [colors[edge_idx(a,b)], colors[edge_idx(b,c)], colors[edge_idx(c,d)], colors[edge_idx(d,a)]]
            if len(set(c1)) == 4:
                return False
            c2 = [colors[edge_idx(a,b)], colors[edge_idx(b,d)], colors[edge_idx(d,c)], colors[edge_idx(c,a)]]
            if len(set(c2)) == 4:
                return False
            c3 = [colors[edge_idx(a,c)], colors[edge_idx(c,b)], colors[edge_idx(b,d)], colors[edge_idx(d,a)]]
            if len(set(c3)) == 4:
                return False
        return True

    # --- Helper: count colors ---
    def _count_colors(colors):
        return len(set(colors))

    # --- Build Gallai recursive coloring (baseline) ---
    def build_gallai():
        dp = [0] * (N + 1)
        split = [0] * (N + 1)
        dp[1] = 0
        dp[2] = 1
        dp[3] = 2
        dp[4] = 4
        for n in range(5, N + 1):
            best_val = 0
            best_k = 1
            for k in range(1, n):
                val = dp[k] + dp[n - k] + 1
                if val > best_val:
                    best_val = val
                    best_k = k
            dp[n] = best_val
            split[n] = best_k

        colors = [0] * NUM_EDGES

        def assign_edge(u, v, c):
            idx = edge_idx(u, v)
            colors[idx] = c

        def build_block(vertices, start_color):
            n = len(vertices)
            if n == 1:
                return start_color
            if n == 2:
                u, v = vertices
                assign_edge(u, v, start_color)
                return start_color + 1
            if n == 3:
                u, v, w = vertices
                assign_edge(u, v, start_color)
                assign_edge(u, w, start_color + 1)
                assign_edge(v, w, start_color + 2)
                return start_color + 3
            if n == 4:
                u, v, w, x = vertices
                A, B, C, D = start_color, start_color+1, start_color+2, start_color+3
                assign_edge(u, v, A)
                assign_edge(w, x, A)
                assign_edge(u, w, B)
                assign_edge(v, x, B)
                assign_edge(u, x, C)
                assign_edge(v, w, D)
                return start_color + 4
            k = split[n]
            v1 = vertices[:k]
            v2 = vertices[k:]
            next_color = build_block(v1, start_color)
            next_color = build_block(v2, next_color)
            cross_color = next_color
            for u in v1:
                for v in v2:
                    assign_edge(u, v, cross_color)
            return cross_color + 1

        all_vertices = list(range(N))
        total_colors = build_block(all_vertices, 0)
        return colors

    # --- Local search to increase color count ---
    def local_search(colors, max_time):
        """
        Try to increase the number of distinct colors by recoloring individual edges.
        For each edge, try assigning a new color (from 0..119) that doesn't create
        a rainbow 4-cycle. Accept if it increases total distinct color count.
        """
        colors = list(colors)
        end_time = time.time() + max_time
        improved = True
        while improved and time.time() < end_time:
            improved = False
            # Shuffle edges for diversity
            edge_order = list(range(NUM_EDGES))
            random.shuffle(edge_order)
            for eidx in edge_order:
                if time.time() >= end_time:
                    break
                i, j = idx_to_edge(eidx)
                old_color = colors[eidx]
                # Find all 4-cycles containing this edge
                # Edge (i,j) is in a 4-cycle with any two other vertices k,l
                # The three cycles on {i,j,k,l} that include edge (i,j):
                #   i-j-k-l-i, i-j-l-k-i, and we need to check all 3 cycles on the quad
                # Actually simpler: for each pair (k,l) with k,l != i,j,
                # check if changing edge (i,j) to new_color creates any rainbow 4-cycle
                # in the 4-vertex set {i,j,k,l}.
                other_vertices = [v for v in range(N) if v != i and v != j]
                valid_new_colors = set()
                # Try colors 0..119
                for new_c in range(120):
                    if new_c == old_color:
                        continue
                    # Check all 4-cycles involving edge (i,j)
                    ok = True
                    for k in range(N):
                        if k == i or k == j:
                            continue
                        for l in range(k+1, N):
                            if l == i or l == j:
                                continue
                            # 4 vertices: i, j, k, l
                            # Three cycles:
                            # Cycle 1: i-j-k-l-i (uses edges ij, jk, kl, li)
                            # Cycle 2: i-j-l-k-i (uses edges ij, jl, lk, ki)
                            # Cycle 3: i-k-j-l-i (uses edges ik, kj, jl, li) -- doesn't use ij directly
                            # Wait, let me reconsider. The three Hamiltonian 4-cycles on {i,j,k,l}:
                            # Fixing i and j as adjacent in the cycle:
                            #   i-j-k-l-i : edges (i,j),(j,k),(k,l),(l,i)
                            #   i-j-l-k-i : edges (i,j),(j,l),(l,k),(k,i)
                            # The third cycle where i and j are NOT adjacent:
                            #   i-k-j-l-i : edges (i,k),(k,j),(j,l),(l,i)
                            # So edge (i,j) appears in cycles 1 and 2 but not cycle 3.
                            # For cycle 3, changing (i,j) doesn't affect it directly,
                            # but we still need to verify it's not rainbow (it wasn't before
                            # and we're only changing (i,j), so if it was valid before,
                            # it remains valid... unless the new color somehow matters.
                            # Actually cycle 3 doesn't use edge (i,j), so it's unaffected.
                            # So we only need to check cycles 1 and 2 for each (k,l).
                            
                            # Cycle 1: i-j-k-l-i
                            c1_colors = {new_c, colors[edge_idx(j,k)], colors[edge_idx(k,l)], colors[edge_idx(l,i)]}
                            if len(c1_colors) == 4:
                                ok = False
                                break
                            # Cycle 2: i-j-l-k-i
                            if ok:
                                c2_colors = {new_c, colors[edge_idx(j,l)], colors[edge_idx(l,k)], colors[edge_idx(k,i)]}
                                if len(c2_colors) == 4:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok:
                        valid_new_colors.add(new_c)
                
                if valid_new_colors:
                    # Pick a color that maximizes distinct color count
                    current_set = set(colors)
                    best_new = None
                    best_gain = 0
                    for nc in valid_new_colors:
                        new_set = current_set - {old_color} | {nc}
                        gain = len(new_set) - len(current_set)
                        # Prefer colors not already in use (gain = 1) over already-used (gain = 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_new = nc
                        elif gain == best_gain and nc not in current_set:
                            best_new = nc
                    if best_new is not None and best_gain >= 0:
                        colors[eidx] = best_new
                        improved = True
        return colors

    # --- Modular arithmetic initialization ---
    def modular_init(a, b, m):
        """c(i,j) = (a*i + b*j) mod m, for i < j"""
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i+1, N):
                colors[edge_idx(i, j)] = (a * i + b * j) % m
        return colors

    # --- Main logic ---
    best_colors = build_gallai()
    best_count = _count_colors(best_colors)
    
    # Try several modular initializations + local search
    random.seed(42)
    m_values = [19, 20, 21, 22, 23, 24]
    ab_pairs = [(1,1), (1,2), (1,3), (2,1), (2,3), (3,1), (3,2), (3,4), (1,4), (2,5), (3,5), (4,1), (4,3), (5,1), (5,2)]
    
    for m in m_values:
        if time.time() > deadline:
            break
        for (a, b) in ab_pairs:
            if time.time() > deadline:
                break
            init_colors = modular_init(a, b, m)
            # Quick validity check
            if _is_valid(init_colors):
                cnt = _count_colors(init_colors)
                if cnt > best_count:
                    best_colors = init_colors
                    best_count = cnt
            # Local search to improve
            remaining = deadline - time.time()
            if remaining > 2:
                searched = local_search(init_colors, min(remaining - 1, 10))
                if _is_valid(searched):
                    cnt = _count_colors(searched)
                    if cnt > best_count:
                        best_colors = searched
                        best_count = cnt
    
    # Also try local search starting from Gallai coloring
    remaining = deadline - time.time()
    if remaining > 2:
        searched = local_search(best_colors, min(remaining - 1, 15))
        if _is_valid(searched):
            cnt = _count_colors(searched)
            if cnt > best_count:
                best_colors = searched
                best_count = cnt

    # Final validity check
    if not _is_valid(best_colors):
        # Fallback to linear ordering coloring (proven valid, 15 colors)
        best_colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i+1, N):
                best_colors[edge_idx(i, j)] = j

    return best_colors

if __name__ == "__main__":
    result = construct()
    print(f"Colors used: {len(set(result))}")
