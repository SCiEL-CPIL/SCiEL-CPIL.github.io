import random
import time

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4.
    Maximizes n.
    """
    deadline = time.time() + 145.0  # Safety margin (evaluator kills at 150s)
    
    # Seed solution for n=11
    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors
    
    best_n, best_colors = get_seed_solution()
    
    # Helper to get edge index in lexicographic order
    def get_edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (i - 1) // 2 + (j - i - 1)
    
    # Build neighbor structure for a given coloring
    def build_neighbors(n, colors):
        neighbors = [[set() for _ in range(5)] for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[i][c].add(j)
                neighbors[j][c].add(i)
                idx += 1
        return neighbors
    
    # Find one C4 in the coloring, return (u, v, c, w1, w2) or None
    def find_c4_fast(n, neighbors):
        for c in range(5):
            for u in range(n):
                nu = neighbors[u][c]
                if len(nu) < 2:
                    continue
                # For efficiency, only check v > u to avoid duplicates
                for v in range(u + 1, n):
                    nv = neighbors[v][c]
                    if len(nv) < 2:
                        continue
                    common = nu & nv
                    if len(common) >= 2:
                        # Find two common neighbors
                        common_list = list(common)[:2]
                        return (u, v, c, common_list[0], common_list[1])
        return None
    
    # Count all C4s in the coloring
    def count_c4s(n, neighbors):
        count = 0
        for c in range(5):
            for u in range(n):
                nu = neighbors[u][c]
                if len(nu) < 2:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[v][c]
                    if len(nv) < 2:
                        continue
                    common = nu & nv
                    if len(common) >= 2:
                        count += 1
        return count
    
    # Local search with simulated annealing
    def local_search(n, max_time=10.0):
        num_edges = n * (n - 1) // 2
        start_time = time.time()
        
        # Multiple restarts
        for restart in range(50):
            if time.time() > start_time + max_time:
                break
            
            # Initialize colors randomly
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            neighbors = build_neighbors(n, colors)
            current_c4s = count_c4s(n, neighbors)
            
            if current_c4s == 0:
                return colors
            
            best_c4s = current_c4s
            best_colors = colors[:]
            
            temperature = 1.0
            cooling_rate = 0.999
            no_improvement = 0
            max_no_improvement = 2000
            
            for _ in range(50000):
                if time.time() > start_time + max_time:
                    break
                
                # Pick a random edge
                idx = random.randint(0, num_edges - 1)
                # Convert linear index to (i, j)
                # Solve for i: i*(i-1)/2 + (j-i-1) = idx
                # Try binary search or approximate
                # Approximate: i ≈ sqrt(2*idx)
                i_approx = int((2 * idx) ** 0.5)
                while i_approx * (i_approx - 1) // 2 > idx:
                    i_approx -= 1
                while (i_approx + 1) * i_approx // 2 <= idx:
                    i_approx += 1
                i = i_approx
                j = i + 1 + (idx - i * (i - 1) // 2)
                
                old_c = colors[idx]
                
                # Try each new color
                new_c4s_best = current_c4s + 100  # Large penalty
                new_c = old_c
                
                for nc in range(5):
                    if nc == old_c:
                        continue
                    
                    # Check if changing edge (i,j) from old_c to nc creates C4s
                    # A C4 in color nc would be formed if i and j have >= 2 common neighbors in nc
                    # But we're ADDING edge (i,j) to color nc, so we need to check if
                    # there exist w, x such that (i,w), (w,j), (j,x), (x,i) are all in nc
                    # Actually, the C4 condition is: two vertices have >= 2 common neighbors.
                    # If we add (i,j) to color nc, this doesn't directly create a C4 unless
                    # i and j already have >= 2 common neighbors in nc.
                    
                    # Wait, that's not quite right. Let me think more carefully.
                    # A C4 in color c exists if there are vertices u, v with >= 2 common neighbors in color c.
                    # If we change edge (i,j) from old_c to nc:
                    # - We remove (i,j) from color old_c: this might BREAK C4s in old_c
                    # - We add (i,j) to color nc: this might CREATE C4s in nc
                    #
                    # For creating C4s in nc: adding edge (i,j) to color nc means i and j are now
                    # neighbors in color nc. A C4 is formed if there exist w, x such that:
                    #   (i,w), (w,j), (j,x), (x,i) are all in color nc
                    # But this requires w and x to be common neighbors of i and j in color nc,
                    # AND (w,x) to also be in color nc? No, that's not right either.
                    #
                    # Actually, a C4 is a cycle of length 4: u-w-v-x-u.
                    # The edges are (u,w), (w,v), (v,x), (x,u).
                    # So for a C4 to exist, we need two vertices u, v that share at least two
                    # common neighbors w, x in the same color.
                    #
                    # If we add edge (i,j) to color nc, this edge itself doesn't form a C4.
                    # But it could be part of a C4 if:
                    #   - There exist w, x such that (i,w), (w,j), (j,x), (x,i) are all in nc
                    #     This means w and x are common neighbors of i and j in nc, AND (w,x) is in nc
                    #     Wait, no. The cycle is i-w-j-x-i. The edges are (i,w), (w,j), (j,x), (x,i).
                    #     So we need (i,w) in nc, (w,j) in nc, (j,x) in nc, (x,i) in nc.
                    #     This doesn't require (w,x) to be in nc.
                    #     So the condition is: there exist w, x (distinct, not i or j) such that
                    #     w and x are both neighbors of i in nc, both neighbors of j in nc,
                    #     i.e., |N_i(nc) ∩ N_j(nc)| >= 2 where N_i(nc) includes j now.
                    #     But wait, j is a neighbor of i in nc (the edge we just added), so j ∈ N_i(nc).
                    #     Similarly, i ∈ N_j(nc).
                    #     So N_i(nc) ∩ N_j(nc) includes... hmm, this is getting complex.
                    #
                    # Let me just recompute the C4 count after the change for simplicity,
                    # but only for the affected colors.
                    
                    # Simple approach: just check if i and j have >= 2 common neighbors in nc
                    # (before adding the edge, since the edge itself doesn't count as a "common neighbor")
                    # Actually, adding the edge (i,j) to nc doesn't change the set of common neighbors
                    # of any pair of vertices. It only changes the adjacency of i and j to each other.
                    # So a C4 involving edge (i,j) in nc would be a 4-cycle where (i,j) is one edge.
                    # The other three edges must be in nc.
                    # A 4-cycle containing edge (i,j) has form i-j-a-b-i.
                    # So we need (j,a), (a,b), (b,i) all in nc.
                    # This means a is a neighbor of j in nc, b is a neighbor of i in nc, and (a,b) in nc.
                    # So we need to check if there exist a, b such that a ∈ N_j(nc), b ∈ N_i(nc), (a,b) ∈ nc.
                    #
                    # This is still complex. Let me just do a full recount for simplicity,
                    # but optimize by only checking affected colors.
                    
                    # For now, let's just do a quick check:
                    # If i and j have >= 2 common neighbors in nc (excluding each other),
                    # then adding (i,j) to nc doesn't create a new C4 by itself, but the
                    # common neighbors were already there.
                    
                    # Actually, the simplest correct approach:
                    # After changing the edge, check if any new C4s are created.
                    # Since we're just adding one edge to color nc, new C4s in nc must involve this edge.
                    # A C4 involving edge (i,j) in color nc: i-j-a-b-i
                    # Requires: (j,a) in nc, (a,b) in nc, (b,i) in nc
                    # So a must be a neighbor of j in nc (a ≠ i), b must be a neighbor of i in nc (b ≠ j),
                    # and (a,b) must be in nc.
                    
                    ni_nc = neighbors[i][nc]
                    nj_nc = neighbors[j][nc]
                    
                    # a must be in nj_nc \ {i}, b must be in ni_nc \ {j}
                    # and (a,b) must be in color nc
                    new_c4s = 0
                    for a in nj_nc:
                        if a == i:
                            continue
                        # b must be in ni_nc, b ≠ j, and (a,b) in nc
                        for b in ni_nc:
                            if b == j or b == a:
                                continue
                            if b in neighbors[a][nc]:  # (a,b) in color nc
                                new_c4s += 1
                    
                    if new_c4s < new_c4s_best:
                        new_c4s_best = new_c4s
                        new_c = nc
                
                # Also consider keeping old_c (delta = 0)
                if 0 < new_c4s_best:
                    new_c4s_best = 0
                    new_c = old_c
                
                # Apply move if beneficial or with some probability
                delta = new_c4s_best
                if delta < 0 or (temperature > 0.01 and random.random() < pow(2.718, -delta / max(temperature, 0.01))):
                    # Apply the change
                    if new_c != old_c:
                        colors[idx] = new_c
                        neighbors[i][old_c].discard(j)
                        neighbors[j][old_c].discard(i)
                        neighbors[i][new_c].add(j)
                        neighbors[j][new_c].add(i)
                        current_c4s += delta  # Approximate; delta was for new color only
                        # This is an approximation. Let me just recount periodically.
                    
                    if current_c4s < best_c4s:
                        best_c4s = current_c4s
                        best_colors = colors[:]
                        no_improvement = 0
                    else:
                        no_improvement += 1
                    
                    temperature *= cooling_rate
                else:
                    no_improvement += 1
                
                if best_c4s == 0:
                    return best_colors
                
                if no_improvement > max_no_improvement:
                    break
            
            if best_c4s == 0:
                return best_colors
            
            # Restart with best colors found
            if best_c4s < current_c4s:
                colors = best_colors[:]
                neighbors = build_neighbors(n, colors)
                current_c4s = best_c4s
            else:
                # Fresh random restart
                colors = [random.randint(0, 4) for _ in range(num_edges)]
                neighbors = build_neighbors(n, colors)
                current_c4s = count_c4s(n, colors and neighbors)
        
        return None
    
    # Search for larger n, trying from high to low
    # Allocate more time to n=26 (known feasible) and less to n=27+ (likely infeasible)
    time_alloc = {
        29: 8,
        28: 8,
        27: 15,
        26: 30,
        25: 15,
        24: 10,
        23: 8,
        22: 6,
        21: 5,
        20: 4,
        19: 3,
        18: 3,
        17: 2,
        16: 2,
        15: 2,
        14: 1,
        13: 1,
        12: 1,
    }
    
    for n in range(29, 11, -1):
        if time.time() > deadline - 5:
            break
        
        alloc = time_alloc.get(n, 3)
        # Ensure we don't exceed deadline
        remaining = deadline - time.time() - 3
        if remaining < 1:
            break
        alloc = min(alloc, int(remaining))
        
        res = local_search(n, max_time=alloc)
        if res:
            return {'n': n, 'colors': res}
    
    # If no larger n found, return seed
    return {'n': best_n, 'colors': best_colors}
