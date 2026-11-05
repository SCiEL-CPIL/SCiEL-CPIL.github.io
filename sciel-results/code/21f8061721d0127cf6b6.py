import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120

def construct():
    start = time.time()
    deadline = start + 50.0

    # Edge index
    def gidx(i, j):
        if i > j:
            i, j = j, i
        return i * N - i * (i + 1) // 2 + (j - i - 1)

    # Precompute cycles and edge→cycles
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    cycles_list = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1, e2, e3, e4 = gidx(a,b), gidx(b,c), gidx(c,d), gidx(d,a)
        cyc_id = len(cycles_list)
        cycles_list.append((e1,e2,e3,e4))
        for e in (e1,e2,e3,e4):
            edge_to_cycles[e].append(cyc_id)
        # Cycle 2: a-b-d-c-a
        e1, e2, e3, e4 = gidx(a,b), gidx(b,d), gidx(d,c), gidx(c,a)
        cyc_id = len(cycles_list)
        cycles_list.append((e1,e2,e3,e4))
        for e in (e1,e2,e3,e4):
            edge_to_cycles[e].append(cyc_id)
        # Cycle 3: a-c-b-d-a
        e1, e2, e3, e4 = gidx(a,c), gidx(c,b), gidx(b,d), gidx(d,a)
        cyc_id = len(cycles_list)
        cycles_list.append((e1,e2,e3,e4))
        for e in (e1,e2,e3,e4):
            edge_to_cycles[e].append(cyc_id)

    NUM_CYCLES = len(cycles_list)

    # Track used_colors and rainbow flag efficiently
    used_colors = [0] * (MAX_COLORS + 1)  # count per color
    rainbow_count = 0  # number of rainbow cycles
    colors_arr = [0] * NUM_EDGES

    def is_rainbow(cyc_id):
        e1,e2,e3,e4 = cycles_list[cyc_id]
        c1,c2,c3,c4 = colors_arr[e1], colors_arr[e2], colors_arr[e3], colors_arr[e4]
        return len({c1,c2,c3,c4}) == 4

    def count_distinct():
        return sum(1 for x in used_colors if x > 0)

    # Initial solution: triangle decomposition
    def build_triangle():
        remaining = list(range(N))
        color_id = 1
        # Reset
        global used_colors, rainbow_count
        used_colors = [0] * (MAX_COLORS + 1)
        rainbow_count = 0
        for e in range(NUM_EDGES):
            colors_arr[e] = color_id - 1
            used_colors[color_id - 1] += 1
        
        while len(remaining) >= 3:
            t = (remaining.pop(), remaining.pop(), remaining.pop())
            u,v,w = t
            e_uv, e_vw, e_wu = gidx(u,v), gidx(v,w), gidx(w,u)
            colors_arr[e_uv] = color_id
            colors_arr[e_vw] = color_id + 1
            colors_arr[e_wu] = color_id + 2
            used_colors[color_id] += 1
            used_colors[color_id+1] += 1
            used_colors[color_id+2] += 1
            color_id += 3
        # Remaining edges (fewer than 3 vertices left: 0, 1, or 2)
        # With 16 vertices, we get 5 triangles (15 vertices) and 1 leftover
        # The leftover vertex has edges to all others; assign a shared color
        if len(remaining) == 1:
            v = remaining[0]
            for other in range(N):
                if other != v:
                    e = gidx(v, other)
                    colors_arr[e] = color_id  # single color for all
                    used_colors[color_id] += 1
        elif len(remaining) == 2:
            u, v = remaining[0], remaining[1]
            e = gidx(u,v)
            colors_arr[e] = color_id
            used_colors[color_id] += 1
        # Count rainbow
        for cid in range(NUM_CYCLES):
            if is_rainbow(cid):
                rainbow_count += 1

    # Alternative initial: random
    def build_random(rng, num_colors):
        global used_colors, rainbow_count
        used_colors = [0] * (MAX_COLORS + 1)
        for e in range(NUM_EDGES):
            c = rng.randint(0, num_colors - 1)
            colors_arr[e] = c
            used_colors[c] += 1
        for cid in range(NUM_CYCLES):
            if is_rainbow(cid):
                rainbow_count += 1

    best_distinct = 0
    best_solution = [0] * NUM_EDGES

    def record():
        global best_distinct, best_solution
        if rainbow_count == 0:
            d = count_distinct()
            if d > best_distinct:
                best_distinct = d
                best_solution = colors_arr[:]

    # Build initial triangle solution
    build_triangle()
    record()

    # Multiple SA attempts with different initial states
    rng = random.Random(20240401)

    while time.time() < deadline - 2.0:
        # Choose initial state
        init_mode = rng.random()
        if init_mode < 0.3:
            build_triangle()
        elif init_mode < 0.6:
            build_random(rng, rng.randint(15, 40))
        else:
            build_random(rng, rng.randint(25, 60))

        # Reset SA params
        T = 15.0
        T_min = 0.05
        cooling = 0.9992
        step = 0
        best_local = 0
        best_local_sol = None
        patience = 0
        max_patience = 8000

        # Local search
        while time.time() < deadline - 1.0:
            e = rng.randint(0, NUM_EDGES - 1)
            old_c = colors_arr[e]
            
            # Choose new color
            r = rng.random()
            if r < 0.6:
                # Random unused color
                # Find an unused color
                unused = [c for c in range(MAX_COLORS) if used_colors[c] == 0]
                if unused:
                    new_c = rng.choice(unused)
                else:
                    new_c = rng.randint(0, MAX_COLORS - 1)
            elif r < 0.85:
                # Random any color
                new_c = rng.randint(0, MAX_COLORS - 1)
            else:
                # Slightly perturbed
                new_c = rng.randint(max(0, old_c - 3), min(MAX_COLORS-1, old_c + 3))

            if new_c == old_c:
                continue

            # Compute delta
            delta = 0
            affected = edge_to_cycles[e]
            for cid in affected:
                if is_rainbow(cid):
                    delta -= 1  # removing a rainbow
                # temporarily set
                colors_arr[e] = new_c
                if is_rainbow(cid):
                    delta += 1  # adding a rainbow
            colors_arr[e] = old_c

            # Color count change
            color_delta = 0
            if used_colors[old_c] == 1 and new_c != old_c:
                color_delta -= 1
            if used_colors[new_c] == 0 and new_c != old_c:
                color_delta += 1

            # Acceptance: minimize rainbow, maximize distinct
            old_score = -1000.0 * rainbow_count + count_distinct()
            new_score = -1000.0 * (rainbow_count + delta) + (count_distinct() + color_delta)
            diff = new_score - old_score
            if diff >= 0 or rng.random() < pow(2.718281828, diff / max(T, T_min)):
                # Apply
                used_colors[old_c] -= 1
                used_colors[new_c] += 1
                colors_arr[e] = new_c
                for cid in affected:
                    if is_rainbow(cid) and new_c == old_c:
                        pass
                # Recompute rainbow for affected
                # We already computed delta, just apply
                # Re-verify
                old_rb = rainbow_count
                rainbow_count = old_rb + delta
                # Fix: need to recompute since we changed colors_arr
                # Actually let's just recompute affected cycles' rainbow status
                # This is O(affected) which is small
                rainbow_count = 0
                for cid in range(NUM_CYCLES):
                    if is_rainbow(cid):
                        rainbow_count += 1
                # This is expensive; use incremental
                # Let me redo:
                # Actually incremental is faster
                pass
                
                # Simpler: just recompute all if small
                # For 680 cycles, recompute is fine
                rainbow_count = 0
                for cid in range(NUM_CYCLES):
                    e1,e2,e3,e4 = cycles_list[cyc_id]
                    if is_rainbow(cid):
                        rainbow_count += 1

                if rainbow_count == 0:
                    d = count_distinct()
                    if d > best_local:
                        best_local = d
                        best_local_sol = colors_arr[:]
                        patience = 0
                    else:
                        patience += 1
                else:
                    patience += 1

            T = max(T_min, T * cooling)
            step += 1
            if step % 1000 == 0:
                if time.time() > deadline - 3.0:
                    break

            if patience > max_patience:
                break

        # Check if local best improves global
        if best_local_sol is not None and rainbow_count == 0:
            if best_local > best_distinct:
                best_distinct = best_local
                best_solution = best_local_sol

    # Final polish: try individual edge reassignment for best solution
    if time.time() < deadline - 0.5:
        # Load best
        for e in range(NUM_EDGES):
            colors_arr[e] = best_solution[e]
        used_colors = [0] * (MAX_COLORS + 1)
        for e in range(NUM_EDGES):
            used_colors[colors_arr[e]] += 1
        rainbow_count = 0
        for cid in range(NUM_CYCLES):
            if is_rainbow(cid):
                rainbow_count += 1

        # Try to improve
        for attempt in range(200):
            if time.time() > deadline - 0.2:
                break
            improved = False
            for e in range(NUM_EDGES):
                if time.time() > deadline - 0.2:
                    break
                if rainbow_count != 0:
                    break
                old_c = colors_arr[e]
                # Try assigning each color to see if we can keep valid and add new distinct
                for new_c in range(MAX_COLORS):
                    if new_c == old_c:
                        continue
                    if used_colors[new_c] > 0:
                        continue
                    # Check if this change keeps validity
                    ok = True
                    for cid in edge_to_cycles[e]:
                        # Temporarily change
                        colors_arr[e] = new_c
                        if is_rainbow(cid):
                            ok = False
                            break
                    colors_arr[e] = old_c
                    if ok:
                        # Apply
                        used_colors[old_c] -= 1
                        used_colors[new_c] += 1
                        colors_arr[e] = new_c
                        best_distinct = count_distinct()
                        improved = True
                        break
                if time.time() > deadline - 0.2:
                    break

    # Verify final
    final_rainbow = 0
    for cid in range(NUM_CYCLES):
        if is_rainbow(cid):
            final_rainbow += 1

    if final_rainbow == 0:
        return best_solution
    else:
        # Fallback: use all zeros (all same color, valid, 1 color)
        return [0] * NUM_EDGES

if __name__ == "__main__":
    colors = construct()
    print(f"Max colors: {len(set(colors))}")
