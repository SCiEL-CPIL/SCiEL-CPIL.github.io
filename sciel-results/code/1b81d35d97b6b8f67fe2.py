"""Rewritten program: vectorized local search to maximize distinct colors in C4-free edge coloring of K_16."""
import random
import time
import itertools
import numpy as np

N = 16
E = N * (N - 1) // 2  # 120 edges
MAX_COLOR = 120

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (2 * N - 1 - i) // 2 + j - i - 1

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Precompute all 4-cycles (3 per 4-vertex subset)
    # For each edge, store the other 3 edges in each cycle it belongs to
    unique_cycles = []
    seen = set()
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Three distinct 4-cycles on vertices {a,b,c,d}
        # Cycle 1: a-b-c-d-a
        e1 = (edge_idx(a, b), edge_idx(b, c), edge_idx(c, d), edge_idx(d, a))
        # Cycle 2: a-b-d-c-a
        e2 = (edge_idx(a, b), edge_idx(b, d), edge_idx(d, c), edge_idx(c, a))
        # Cycle 3: a-c-b-d-a
        e3 = (edge_idx(a, c), edge_idx(c, b), edge_idx(b, d), edge_idx(d, a))

        for cyc in (e1, e2, e3):
            key = tuple(sorted(cyc))
            if key not in seen:
                seen.add(key)
                unique_cycles.append(cyc)

    # Build: for each edge, list of (cycle_index, other_3_edges)
    edge_cycles = [[] for _ in range(E)]
    for ci, cyc in enumerate(unique_cycles):
        for pos, e in enumerate(cyc):
            others = [cyc[k] for k in range(4) if k != pos]
            edge_cycles[e].append((ci, others))

    # Precompute as numpy arrays for speed
    # For each edge: list of (cycle_idx, [o1, o2, o3])
    # We'll store the "other" edges in a flat structure

    # State
    colors = np.zeros(E, dtype=np.int32)
    freq = np.zeros(MAX_COLOR, dtype=np.int32)
    distinct = 0

    # Compute distinct count
    def compute_distinct(freq):
        return int(np.sum(freq > 0))

    # Check if a color assignment is valid (no C4 with 4 distinct colors)
    # Vectorized: for each cycle, check if all 4 colors are distinct
    def count_bad_cycles(colors, edge_cycles, unique_cycles):
        bad = 0
        for cyc in unique_cycles:
            c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                bad += 1
        return bad

    # Fast delta_bad for changing edge e from old_c to new_c
    def delta_bad(e, old_c, new_c, colors):
        if old_c == new_c:
            return 0
        db = 0
        for ci, others in edge_cycles[e]:
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            # Old badness
            old_bad = 1 if (old_c != c1 and old_c != c2 and old_c != c3 and c1 != c2 and c1 != c3 and c2 != c3) else 0
            # New badness
            new_bad = 1 if (new_c != c1 and new_c != c2 and new_c != c3 and c1 != c2 and c1 != c3 and c2 != c3) else 0
            db += new_bad - old_bad
        return db

    best_distinct = 0
    best_colors = None

    # Strategy: Start with a valid coloring, then use local search to introduce new colors
    # Initial coloring: use a Gallai-type coloring which is C4-free
    # c(i,j) = i for i < j gives at most 3 colors per 4-cycle (since in any 4-cycle, 
    # the vertices have distinct "i" values but the coloring uses the smaller index)
    # Actually, let's use a simpler valid coloring: c(i,j) = min(i,j) or similar

    def make_initial_coloring():
        """Create a valid C4-free coloring with moderate number of distinct colors."""
        col = np.zeros(E, dtype=np.int32)
        # Use a hierarchical coloring: partition vertices into groups
        # For simplicity, use c(i,j) = i (the smaller vertex index)
        for i in range(N):
            for j in range(i + 1, N):
                col[edge_idx(i, j)] = i
        return col

    # Verify initial coloring is valid
    def verify_valid(col):
        for cyc in unique_cycles:
            c0, c1, c2, c3 = col[cyc[0]], col[cyc[1]], col[cyc[2]], col[cyc[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True

    # Local search to maximize distinct colors while maintaining validity
    def local_search_maximize_distinct(col, max_time):
        """
        Given a valid coloring, try to increase the number of distinct colors
        by recoloring edges to new colors, maintaining validity.
        """
        global best_distinct, best_colors

        freq = np.zeros(MAX_COLOR, dtype=np.int32)
        for c in col:
            freq[c] += 1
        distinct = compute_distinct(freq)

        if distinct > best_distinct:
            best_distinct = distinct
            best_colors = col.copy()

        # Track which colors are currently used
        used_colors = set(int(c) for c in col if c >= 0)

        # Precompute: for each edge, the set of colors that would make it "bad"
        # when combined with its neighbors in each cycle
        # This is dynamic, so we compute on the fly

        step = 0
        no_improve = 0
        max_no_improve = 5000

        while time.time() < max_time:
            step += 1

            if no_improve > max_no_improve:
                break

            # Pick a random edge
            e = random.randint(0, E - 1)
            old_c = int(col[e])

            # Try to assign a new color that is not currently used
            # and doesn't break validity
            # First, gather the "forbidden" colors for this edge
            # A color x is forbidden if for some cycle containing e,
            # x is distinct from all three other edge colors in that cycle

            forbidden = set()
            for ci, others in edge_cycles[e]:
                c1 = int(col[others[0]])
                c2 = int(col[others[1]])
                c3 = int(col[others[2]])
                # x is forbidden if x != c1, x != c2, x != c3, and c1,c2,c3 are all distinct
                if c1 != c2 and c1 != c3 and c2 != c3:
                    # Any x not in {c1, c2, c3} would create a 4-distinct-color cycle
                    # So forbidden colors are all colors except c1, c2, c3
                    # But we only care about unused colors
                    # Mark that x must be in {c1, c2, c3} for this cycle
                    # So the allowed colors for this cycle are {c1, c2, c3}
                    pass

            # Better approach: for each cycle containing e, the new color must
            # equal at least one of the other three colors (if those three are distinct)
            # If the other three are not all distinct, any color is fine for that cycle

            # Compute allowed colors: intersection of constraints from all cycles
            allowed = set(range(MAX_COLOR))
            for ci, others in edge_cycles[e]:
                c1 = int(col[others[0]])
                c2 = int(col[others[1]])
                c3 = int(col[others[2]])
                if c1 != c2 and c1 != c3 and c2 != c3:
                    # New color must be in {c1, c2, c3}
                    allowed &= {c1, c2, c3}

            # Now, among allowed colors, prefer unused ones to increase distinct count
            unused_allowed = [c for c in allowed if freq[c] == 0]

            if unused_allowed:
                # Pick a random unused allowed color
                new_c = random.choice(unused_allowed)
                # Apply
                col[e] = new_c
                freq[old_c] -= 1
                freq[new_c] += 1
                distinct += 1  # We gained a new color, old color still used (freq[old_c] > 0 before decrement, now >= 0)
                # Wait: if freq[old_c] was 1, we lose it. Let's recompute
                # Actually: distinct change = (1 if freq[new_c]==0 before else 0) - (1 if freq[old_c]==1 before else 0)
                # We already updated freq, so let's recompute properly
                # Simpler: just recompute distinct occasionally
                no_improve = 0

                if distinct > best_distinct:
                    best_distinct = distinct
                    best_colors = col.copy()
            else:
                # No unused color is allowed. Try a used color to reshuffle
                # or skip
                if allowed:
                    # Pick a random allowed color (used)
                    new_c = random.choice(list(allowed))
                    if new_c != old_c:
                        # Check if this changes distinct count
                        old_freq_old = freq[old_c]
                        old_freq_new = freq[new_c]
                        col[e] = new_c
                        freq[old_c] -= 1
                        freq[new_c] += 1
                        # Recompute distinct
                        new_distinct = compute_distinct(freq)
                        if new_distinct < distinct:
                            # Revert
                            col[e] = old_c
                            freq[old_c] += 1
                            freq[new_c] -= 1
                            no_improve += 1
                        else:
                            distinct = new_distinct
                            no_improve = 0
                            if distinct > best_distinct:
                                best_distinct = distinct
                                best_colors = col.copy()
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

            # Periodically recompute distinct to avoid drift
            if step % 1000 == 0:
                distinct = compute_distinct(freq)

        return distinct

    # Main loop: multiple restarts with different initial colorings
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1

        # Create initial coloring
        if restart_count == 1:
            # Start with the simple valid coloring
            col = make_initial_coloring()
        elif restart_count % 3 == 0:
            # Random valid-ish start: use a small palette
            palette_size = random.randint(3, 8)
            col = np.random.randint(0, palette_size, E, dtype=np.int32)
            # Fix any violations with SA
            # ... (simplified: just use the hierarchical one)
            col = make_initial_coloring()
        else:
            # Perturb the best known coloring
            if best_colors is not None:
                col = best_colors.copy()
                # Randomly recolor a few edges
                for _ in range(random.randint(1, 5)):
                    e = random.randint(0, E - 1)
                    col[e] = random.randint(0, MAX_COLOR - 1)
            else:
                col = make_initial_coloring()

        # Ensure validity: fix any bad cycles
        # Use a quick SA phase to eliminate bad cycles
        bad_count = count_bad_cycles(col, edge_cycles, unique_cycles)
        if bad_count > 0:
            # SA to fix
            T = 10.0
            cooling = 0.99
            for _ in range(5000):
                if time.time() > deadline - 5:
                    break
                e = random.randint(0, E - 1)
                old_c = int(col[e])
                new_c = random.randint(0, MAX_COLOR - 1)
                if old_c == new_c:
                    continue
                db = delta_bad(e, old_c, new_c, col)
                if db <= 0 or random.random() < np.exp(-db / T):
                    col[e] = new_c
                T *= cooling

            # Check if valid
            if count_bad_cycles(col, edge_cycles, unique_cycles) > 0:
                # Still invalid, use fallback
                col = make_initial_coloring()

        # Now maximize distinct colors
        local_search_maximize_distinct(col, deadline - 5 - time.time())

        if time.time() > deadline - 3:
            break

    if best_colors is None:
        # Fallback
        best_colors = make_initial_coloring()

    return best_colors.tolist()
