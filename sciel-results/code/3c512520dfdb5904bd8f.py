# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = 49
    total = q * q  # 2401
    K = 3  # alpha^2 = 3

    def idx_of(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    def mul(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*K) % p, (a[0]*b[1] + a[1]*b[0]) % p)

    def add(a, b):
        return ((a[0]+b[0]) % p, (a[1]+b[1]) % p)

    def sub(a, b):
        return ((a[0]-b[0]) % p, (a[1]-b[1]) % p)

    # Build all 50 line directions: (1,m) for m in F49, plus (0,1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1,0),(ma,mb)))
    directions.append(((0,1),(0,0)))

    # Build all lines as bitsets (integers with 2401 bits)
    # A line through c with direction v: {c + t*v : t in F49}
    all_lines = []
    for vx, vy in directions:
        if vx[0] == 1:
            m = vy
            # Line: y = m*x + c, for each c in F49
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    bits = 0
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = mul(m, x)
                            y = add(mx, c)
                            idx = idx_of(xa, xb, y[0], y[1])
                            bits |= (1 << idx)
                    all_lines.append(bits)
        else:
            # Vertical lines: x = c
            for ca in range(p):
                for cb in range(p):
                    bits = 0
                    for ya in range(p):
                        for yb in range(p):
                            idx = idx_of(ca, cb, ya, yb)
                            bits |= (1 << idx)
                    all_lines.append(bits)

    num_lines = len(all_lines)  # should be 50

    # For each point, find which lines pass through it
    point_lines = [[] for _ in range(total)]
    for li, bits in enumerate(all_lines):
        # Iterate set bits
        b = bits
        while b:
            lb = b & (-b)
            idx = lb.bit_length() - 1
            point_lines[idx].append(li)
            b ^= lb

    # Verify: each point should be on exactly 50 lines
    # (49 with direction (1,m) + 1 vertical)

    start_time = time.time()
    deadline = start_time + 550.0  # safety margin before 600s kill

    def is_valid(S_bits):
        """Check if S is a blocking set: every line has at least one point in S."""
        for bits in all_lines:
            if (S_bits & bits) == 0:
                return False
        return True

    def score(S_bits):
        """Number of points in S."""
        return S_bits.bit_count()

    # ===== Generate diverse algebraic seed structures =====
    # Hermitian unital: Tr(x) + N(y) = 0  (2520 points)
    # Also try: Tr(x) + c*N(y) = 0 for various c
    # Also try: N(x) + Tr(y) = 0, etc.
    # Also try random dense sets

    def trace(a):
        return (2 * a[0]) % p

    def norm(a):
        return (a[0]*a[0] - a[1]*a[1]*K) % p

    def unital_bits(coeff_x, coeff_y, const):
        """Points where coeff_x*Tr(x) + coeff_y*N(y) + const = 0"""
        bits = 0
        for xa in range(p):
            for xb in range(p):
                x = (xa, xb)
                for ya in range(p):
                    for yb in range(p):
                        y = (ya, yb)
                        if (coeff_x * trace(x) + coeff_y * norm(y) + const) % p == 0:
                            idx = idx_of(xa, xb, ya, yb)
                            bits |= (1 << idx)
        return bits

    def random_dense_bits(fraction):
        bits = 0
        for i in range(total):
            if random.random() < fraction:
                bits |= (1 << i)
        return bits

    def random_sparse_bits(fraction):
        bits = 0
        for i in range(total):
            if random.random() < fraction:
                bits |= (1 << i)
        return bits

    # Seed generators: produce (initial_S_bits, description)
    def generate_seeds():
        seeds = []
        # Hermitian unital variants
        for cx in range(1, p):
            for cy in range(1, p):
                for c in range(p):
                    bits = unital_bits(cx, cy, c)
                    seeds.append(bits)
        # N(x) + Tr(y) = 0 type
        for cx in range(1, p):
            for cy in range(1, p):
                for c in range(p):
                    bits = 0
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            for ya in range(p):
                                for yb in range(p):
                                    y = (ya, yb)
                                    if (cx * norm(x) + cy * trace(y) + c) % p == 0:
                                        idx = idx_of(xa, xb, ya, yb)
                                        bits |= (1 << idx)
                    seeds.append(bits)
        # Random sets of various densities
        for frac in [0.5, 0.6, 0.7, 0.8, 0.9]:
            for _ in range(5):
                seeds.append(random_dense_bits(frac))
        return seeds

    seeds = generate_seeds()
    random.shuffle(seeds)

    best_S_bits = 0
    best_score_val = 0  # complement size = total - |S|

    def try_improve(S_bits, time_budget):
        """Local search: try to remove points from S while maintaining blocking property."""
        nonlocal best_S_bits, best_score_val
        end_time = time.time() + time_budget
        current = S_bits
        current_sz = current.bit_count()

        # Precompute for each point in S, which lines it's the UNIQUE blocker for
        # A point can be removed if no line has it as the only S-point on that line

        def unique_blocker_count(S):
            """For each point in S, count how many lines it uniquely blocks."""
            # For each line, find S-points on it
            unique_count = [0] * total
            for bits in all_lines:
                line_s = S & bits
                if line_s == 0:
                    return None  # invalid
                if line_s & (line_s - 1) == 0:
                    # Exactly one point
                    idx = line_s.bit_length() - 1
                    unique_count[idx] += 1
            return unique_count

        uc = unique_blocker_count(current)
        if uc is None:
            return S_bits, current_sz

        while time.time() < end_time:
            # Find removable points (unique_blocker_count == 0)
            removable = []
            b = current
            while b:
                lb = b & (-b)
                idx = lb.bit_length() - 1
                if uc[idx] == 0:
                    removable.append(idx)
                b ^= lb

            if not removable:
                # Try 2-swap: remove two, add one? Or just try random single removals
                # Try removing a point and see if we can fix by adding another
                # For now, try random point removal and check
                found_move = False
                for _ in range(200):
                    if time.time() >= end_time:
                        break
                    # Pick random point in S
                    b2 = current
                    cnt = current.bit_count()
                    # Pick random position
                    r = random.randint(0, cnt - 1)
                    tmp = b2
                    for i in range(r):
                        lb = tmp & (-tmp)
                        tmp ^= lb
                    lb = tmp & (-tmp)
                    idx_rm = lb.bit_length() - 1
                    new_S = current & ~(1 << idx_rm)
                    if is_valid(new_S):
                        current = new_S
                        current_sz -= 1
                        uc = unique_blocker_count(current)
                        if uc is None:
                            # Shouldn't happen since we checked
                            break
                        found_move = True
                        break
                if not found_move:
                    break

            # Remove one removable point
            idx_rm = random.choice(removable)
            new_S = current & ~(1 << idx_rm)
            current = new_S
            current_sz -= 1

            # Update unique blocker counts incrementally
            # Points that were on lines with idx_rm as unique blocker lose that
            # Lines through idx_rm: for each such line, if it had exactly 1 S-point (idx_rm),
            #   now it has 0 -> invalid! But we checked uc[idx_rm]==0 so no line had idx_rm as unique.
            # However, removing idx_rm may make OTHER points on those lines become unique blockers
            # OR may make lines that had 2 S-points now have 1, making the remaining point unique.
            for li in point_lines[idx_rm]:
                line_bits = all_lines[li]
                line_s = current & line_bits
                if line_s == 0:
                    # Invalid - shouldn't happen
                    break
                if line_s & (line_s - 1) == 0:
                    # Exactly one point now
                    idx = line_s.bit_length() - 1
                    uc[idx] += 1
                # Also, the removed point was on this line; if this line had exactly 2 before,
                # the other point's unique count might change... but we already handled above.
                # Actually we need to be more careful. Let me just recompute for safety on small sets.
            # For correctness, recompute uc periodically
            if current_sz % 50 == 0:
                uc = unique_blocker_count(current)
                if uc is None:
                    # Shouldn't happen
                    break

            if current_sz < best_score_val or best_score_val == 0:
                pass  # best_score_val tracks complement, not S size
            comp = total - current_sz
            if comp > best_score_val:
                best_score_val = comp
                best_S_bits = current

        return current, current_sz

    # Process seeds
    seed_idx = 0
    while seed_idx < len(seeds) and time.time() < deadline - 30:
        S_init = seeds[seed_idx]
        seed_idx += 1

        # First, make S_init valid by adding points to unblocked lines
        # Greedy: for each unblocked line, add a random point from it
        S = S_init
        changed = True
        while changed and time.time() < deadline - 30:
            changed = False
            for bits in all_lines:
                if (S & bits) == 0:
                    # Add a random point from this line
                    b = bits
                    cnt = 0
                    tmp = b
                    while tmp:
                        lb = tmp & (-tmp)
                        tmp ^= lb
                        cnt += 1
                    r = random.randint(0, cnt - 1)
                    tmp = b
                    for i in range(r):
                        lb = tmp & (-tmp)
                        tmp ^= lb
                    lb = tmp & (-tmp)
                    idx = lb.bit_length() - 1
                    S |= (1 << idx)
                    changed = True

        # Now S is valid. Improve by removing points.
        time_per_seed = min(15.0, max(3.0, (deadline - 30 - time.time()) / max(1, len(seeds) - seed_idx)))
        S, sz = try_improve(S, time_per_seed)

    # Final phase: aggressive local search on best found
    if time.time() < deadline - 20 and best_S_bits != 0:
        S = best_S_bits
        end_time = deadline - 15
        # More aggressive: try all single removals, then 2-removals
        while time.time() < end_time:
            # Single removals
            improved = False
            b = S
            while b and time.time() < end_time:
                lb = b & (-b)
                idx = lb.bit_length() - 1
                b ^= lb
                new_S = S & ~(1 << idx)
                if is_valid(new_S):
                    S = new_S
                    improved = True
                    break
            if not improved:
                # Try 2-point removals
                two_improved = False
                b = S
                pts = []
                while b and len(pts) < 200:
                    lb = b & (-b)
                    pts.append(lb.bit_length() - 1)
                    b ^= lb
                for i in range(len(pts)):
                    if time.time() >= end_time:
                        break
                    for j in range(i+1, len(pts)):
                        if time.time() >= end_time:
                            break
                        new_S = S & ~(1 << pts[i]) & ~(1 << pts[j])
                        if is_valid(new_S):
                            S = new_S
                            two_improved = True
                            break
                    if two_improved:
                        break
                if not two_improved:
                    break
            comp = total - S.bit_count()
            if comp > best_score_val:
                best_score_val = comp
                best_S_bits = S

    # Convert best_S_bits to N = complement
    N_indices = []
    for i in range(total):
        if not (best_S_bits >> i) & 1:
            N_indices.append(i)

    points = []
    for idx in N_indices:
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        points.append([[xa, xb], [ya, yb]])

    result = np.array(points, dtype=np.int64)

    # Final validity check
    if len(points) == 0:
        # Fallback: return all points (valid but score 0)
        points = []
        for i in range(total):
            yb = i % p
            ya = (i // p) % p
            xb = (i // (p * p)) % p
            xa = (i // (p * p * p)) % p
            points.append([[xa, xb], [ya, yb]])
        result = np.array(points, dtype=np.int64)

    return result

# EVOLVE-BLOCK-END
