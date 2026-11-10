# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    """
    start_time = time.time()
    DEADLINE = 600  # seconds, safety margin before 630s kill

    p = 7
    q = p * p  # 49

    # alpha^2 = 3 in F_7 (3 is a non-residue mod 7)
    k = 3

    # Field operations in F_49 = F_7[alpha], alpha^2 = 3
    # Element is (a, b) representing a + b*alpha
    def add_f(a, b, c, d):
        return ((a + c) % p, (b + d) % p)

    def mul_f(a, b, c, d):
        # (a + b*alpha)(c + d*alpha) = ac + ad*alpha + bc*alpha + bd*alpha^2
        # = (ac + bd*k) + (ad + bc)*alpha
        return ((a * c + b * d * k) % p, (a * d + b * c) % p)

    def sub_f(a, b, c, d):
        return ((a - c) % p, (b - d) % p)

    def neg_f(a, b):
        return ((-a) % p, (-b) % p)

    def inv_f(a, b):
        # Inverse of a + b*alpha in F_49
        # (a + b*alpha)(a - b*alpha) = a^2 - b^2*alpha^2 = a^2 - b^2*k
        # So inverse is (a - b*alpha) / (a^2 - b^2*k)
        norm = (a * a - b * b * k) % p
        if norm == 0:
            return (0, 0)  # shouldn't happen for nonzero element
        inv_norm = pow(norm, p - 2, p)  # Fermat's little theorem
        return (a * inv_norm % p, (-b * inv_norm) % p)

    # All elements of F_49 as (a, b) pairs
    F49 = [(a, b) for a in range(p) for b in range(p)]
    f49_idx = {(a, b): i for i, (a, b) in enumerate(F49)}

    # Precompute multiplication table
    mul_table = np.zeros((q, q, 2, 2), dtype=np.int64)
    for i, (a, b) in enumerate(F49):
        for j, (c, d) in enumerate(F49):
            ra, rb = mul_f(a, b, c, d)
            mul_table[i, j, 0, 0] = ra
            mul_table[i, j, 0, 1] = rb

    # Precompute addition table
    add_table = np.zeros((q, q, 2, 2), dtype=np.int64)
    for i, (a, b) in enumerate(F49):
        for j, (c, d) in enumerate(F49):
            ra, rb = add_f(a, b, c, d)
            add_table[i, j, 0, 0] = ra
            add_table[i, j, 0, 1] = rb

    # Points in F_49^2: each point is (x, y) where x, y in F_49
    # Index: point_idx = x_idx * q + y_idx
    # Total points: q^2 = 2401

    # Lines in direction (1, m) for m in F_49, and direction (0, 1)
    # A line in direction (1, m) through point (x0, y0): {(x0 + t, y0 + t*m) : t in F_49}
    # A line in direction (0, 1) through point (x0, y0): {(x0, y0 + t) : t in F_49}

    # For each point, we need at least one line through it such that all other points on that line are in N.
    # Equivalently: for each point x, there exists a direction v such that the punctured line (line minus x) is subset of N.
    # This means: for each point x, there exists a direction v such that no other point on the line through x in direction v is in the complement S.

    # Strategy: Greedy set cover to minimize the "hole" set.
    # We want to maximize |S| (complement) such that for every point x in F_49^2, there exists a line L through x with L \ {x} ⊆ N, i.e., L \ {x} ∩ S = ∅.
    # Equivalently: for every point x, there exists a line L through x such that L ∩ S ⊆ {x}.

    # Let's think of it differently. For each point x, define the set of "valid" lines through x: lines L through x such that L ∩ S = {x} (or L ∩ S = ∅ if x not in S, but we need L\{x} ⊆ N which is L\{x} ∩ S = ∅).
    # If x ∈ S: we need a line L through x with L ∩ S = {x}.
    # If x ∉ S: we need a line L through x with L ∩ S = ∅ (since x ∉ S, L\{x} ⊆ N means L ∩ S = ∅).

    # Actually, re-reading: "every point x of F_49^2 (in N or not) must lie on a line {x + t*v : t in F_49}, with v = (1, m) for some m in F_49 or v = (0, 1), whose points other than x all belong to N."
    # So for each x, there exists v such that for all t ≠ 0, x + t*v ∈ N.
    # This is equivalent to: for each x, there exists v such that (line through x in direction v) \ {x} ⊆ N.
    # I.e., for each x, there exists v such that (line through x in direction v) ∩ S ⊆ {x}.

    # Greedy approach: Start with S = ∅. Try to add points to S one at a time, maintaining the validity constraint.
    # When we add a point y to S, we must check that for every point x, there still exists a valid line through x.
    # This is expensive. Instead, let's use the structure.

    # Alternative: The baseline construction gives a valid Nikodym set. Let's start from there and try to improve.
    # Baseline: S = { (x, y) in U : y not in Y } where U is the Hermitian unital and Y is the special set.

    # Let's compute the baseline S first.
    def in_unital(xa, xb, ya, yb):
        # Trace(x) + Norm(y) = 0
        # Trace((xa, xb)) = xa + xb^(-1) ... wait, in F_49/F_7, trace of a + b*alpha is (a + b*alpha) + (a + b*alpha)^7 = (a + b*alpha) + (a + b*alpha^7)
        # Since alpha^7 = alpha (because in F_49, x^49 = x, and alpha^2 = 3, alpha^4 = 9 = 2, alpha^8 = 4, ... let's compute)
        # Actually, in F_{p^2}/F_p, the Frobenius map is x -> x^p. For a + b*alpha, (a + b*alpha)^p = a + b*alpha^p.
        # alpha^p = alpha^7. Since alpha^2 = 3, alpha^4 = 9 mod 7 = 2, alpha^6 = 2*3 = 6 = -1, alpha^7 = -alpha.
        # So (a + b*alpha)^7 = a + b*(-alpha) = a - b*alpha.
        # Trace = (a + b*alpha) + (a - b*alpha) = 2a.
        # Norm = (a + b*alpha)(a - b*alpha) = a^2 - b^2*alpha^2 = a^2 - 3*b^2.
        # Unital: Trace(x) + Norm(y) = 0 => 2*xa + ya^2 - 3*yb^2 = 0 (mod 7)
        return (2 * xa + ya * ya - 3 * yb * yb) % p == 0

    def in_y_special(ya, yb):
        # Y = {(u^2 mod p, v) : u, v in F_p}
        # ya must be a square mod 7
        # Squares mod 7: 0^2=0, 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1
        # So squares are {0, 1, 2, 4}
        return ya in (0, 1, 2, 4)

    # Baseline S
    S_baseline = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if in_unital(xa, xb, ya, yb) and not in_y_special(ya, yb):
                        S_baseline.add(((xa, xb), (ya, yb)))

    # Now, let's try to expand S greedily.
    # For a candidate point y to add to S, we need to check that for every point x, there still exists a valid line.
    # This is O(|F_49^2| * |directions|) per addition, which is 2401 * 50 ≈ 120000 checks.
    # With up to 2401 additions, this could be 2401 * 120000 ≈ 288 million operations, which might be too slow in Python.

    # Let's use a smarter approach. Precompute for each point x the set of directions v such that the line through x in direction v is "compatible" with S.
    # A direction v is compatible with x and S if (line through x in direction v) ∩ S ⊆ {x}.

    # Let's precompute lines.
    # For each direction (1, m) where m in F_49, and for each point (x0, y0), the line is {(x0 + t, y0 + t*m) : t in F_49}.
    # For direction (0, 1), the line is {(x0, y0 + t) : t in F_49}.

    # Let's represent directions as indices. Directions: (1, m) for m in F_49 (49 directions) and (0, 1) (1 direction). Total 50 directions.
    # dir_idx = 0..48 for (1, F49[dir_idx]), dir_idx = 49 for (0, 1).

    # For each point x and direction d, compute the set of other points on the line.
    # line_points[x_idx, d_idx] = list of point indices on the line through x in direction d, excluding x.

    # This is a lot of precomputation. Let's think about memory.
    # 2401 points * 50 directions * 48 other points = 5.7 million entries. Each entry is an int (point index). That's about 23 MB. OK.

    # But building this in Python might be slow. Let's try a different approach.

    # Greedy with lazy checking:
    # Maintain S as a set. For each point x, maintain the count of valid directions (directions d such that line through x in direction d has no points of S other than possibly x).
    # When we add a point y to S, for each line L that contains y, for each point x on L (x ≠ y), decrement the count of valid directions for x (if the direction of L was valid for x).
    # If any point x has 0 valid directions, the addition is invalid.

    # Let's precompute for each point y, the set of (x, d) pairs such that y is on the line through x in direction d (y ≠ x).
    # For a fixed y and direction d, the points x such that y is on the line through x in direction d are: x = y - t*d for t ≠ 0.
    # So for each y, d, there are q-1 = 48 points x.

    # Total pairs: 2401 * 50 * 48 = 5.7 million. This is manageable if we do it efficiently.

    # Let's index points as x_idx * q + y_idx where x_idx, y_idx in 0..48.
    # Point (x, y) has index x_idx * q + y_idx.

    # Let's build the adjacency: for each point y_idx, for each direction d, list of (x_idx, d) where y is on line through x in direction d.
    # Actually, we need: for each point y, the set of (x, d) such that y is on the line through x in direction d, y ≠ x.
    # Then when we add y to S, we need to check all these (x, d) pairs and decrement their validity count.

    # Let's precompute this structure.
    # adj[y_idx] = list of (x_idx, d_idx) for all x, d such that y is on line through x in direction d, x ≠ y.

    # For direction (1, m): line through x in direction (1, m) is {(x + t, y + t*m) : t in F_49}.
    # y is on this line iff y - x = t*(1, m) for some t ≠ 0, i.e., x = y - t*(1, m) for some t ≠ 0.
    # So for each t ≠ 0, x = (y_x - t, y_y - t*m).

    # For direction (0, 1): line through x is {(x_x, x_y + t) : t in F_49}.
    # y is on this line iff y_x = x_x and y_y - x_y = t for some t ≠ 0, i.e., x = (y_x, y_y - t) for t ≠ 0.

    # Let's build adj.
    adj = [[] for _ in range(q * q)]

    for y_xa in range(p):
        for y_xb in range(p):
            for y_ya in range(p):
                for y_yb in range(p):
                    y_idx = f49_idx[(y_xa, y_xb)] * q + f49_idx[(y_ya, y_yb)]
                    for d_idx in range(q):  # directions (1, m)
                        m = F49[d_idx]
                        for t in range(1, q):  # t ≠ 0
                            # x = y - t*(1, m)
                            tx_a, tx_b = F49[t]  # t as element of F_49
                            tm_a, tm_b = mul_f(tx_a, tx_b, m[0], m[1])
                            x_xa, x_xb = sub_f(y_xa, y_xb, tm_a, tm_b)
                            x_ya, x_yb = sub_f(y_ya, y_yb, mul_f(tx_a, tx_b, m[0], m[1])[0], mul_f(tx_a, tx_b, m[0], m[1])[1])
                            # Wait, t*(1, m) = (t, t*m). So x = (y_x - t, y_y - t*m).
                            t_a, t_b = F49[t]
                            tm_a, tm_b = mul_f(t_a, t_b, m[0], m[1])
                            x_xa, x_xb = sub_f(y_xa, y_xb, t_a, t_b)
                            x_ya, x_yb = sub_f(y_ya, y_yb, tm_a, tm_b)
                            x_idx = f49_idx[(x_xa, x_xb)] * q + f49_idx[(x_ya, x_yb)]
                            adj[y_idx].append((x_idx, d_idx))
                    # Direction (0, 1)
                    d_idx = q  # 49
                    for t in range(1, q):
                        t_a, t_b = F49[t]
                        # x = (y_x, y_y - t)
                        x_xa, x_xb = y_xa, y_xb
                        x_ya, x_yb = sub_f(y_ya, y_yb, t_a, t_b)
                        x_idx = f49_idx[(x_xa, x_xb)] * q + f49_idx[(x_ya, x_yb)]
                        adj[y_idx].append((x_idx, d_idx))

    # Now, valid_count[x_idx][d_idx] = 1 if the line through x in direction d is valid (no points of S on it other than possibly x), 0 otherwise.
    # Initially, S = ∅, so all are valid.
    valid_count = np.ones((q * q, q + 1), dtype=np.int32)

    # For each point x, valid_directions_count[x_idx] = number of valid directions.
    valid_dir_count = np.full(q * q, q + 1, dtype=np.int32)

    # Greedy: try to add points to S.
    # We want to maximize |S|. Start with S = ∅ and try to add points.
    # But we need to be careful: adding a point y might invalidate some lines.
    # Let's use a priority queue or just iterate.

    # Actually, let's start with the baseline S and see if we can expand it.
    # Or start from scratch and greedily add points.

    # Let's start from scratch.
    S = set()

    # To make it faster, let's try points in a random order or some heuristic order.
    # For now, let's just iterate through all points and try to add them.
    # This is O(2401 * avg_adj_size) which is 2401 * (50 * 48) ≈ 5.7 million operations. Should be fast enough.

    # Actually, when we add a point y, we need to check all (x, d) in adj[y] and decrement valid_count[x][d].
    # If valid_count[x][d] becomes 0, we decrement valid_dir_count[x].
    # If valid_dir_count[x] becomes 0, the addition is invalid.

    # Let's implement this.
    for y_idx in range(q * q):
        if time.time() - start_time > DEADLINE - 30:
            break

        # Check if adding y is valid
        # Temporarily add y
        affected = []
        valid = True
        for x_idx, d_idx in adj[y_idx]:
            if valid_count[x_idx, d_idx] > 0:
                valid_count[x_idx, d_idx] -= 1
                affected.append((x_idx, d_idx))
                if valid_count[x_idx, d_idx] == 0:
                    valid_dir_count[x_idx] -= 1
                    if valid_dir_count[x_idx] == 0:
                        valid = False
                        break

        if valid:
            S.add(y_idx)
            # Also, we need to consider that y itself might be on lines through other points.
            # But we already handled that via adj[y_idx].
            # However, we also need to check lines that pass through y and other points in S.
            # Wait, the condition is: for each x, there exists a direction d such that line through x in direction d has no points of S other than possibly x.
            # When we add y to S, for each line L that contains y, for each x on L (x ≠ y), the line L is no longer valid for x (if it was valid before and y was the only issue).
            # This is exactly what we're doing with adj[y_idx].
        else:
            # Rollback
            for x_idx, d_idx in affected:
                valid_count[x_idx, d_idx] += 1
                if valid_count[x_idx, d_idx] == 1:
                    valid_dir_count[x_idx] += 1

    # Build the Nikodym set N = F_49^2 \ S
    points = []
    for x_idx in range(q * q):
        if x_idx not in S:
            xa_idx, ya_idx = divmod(x_idx, q)
            xa, xb = F49[xa_idx]
            ya, yb = F49[ya_idx]
            points.append([[xa, xb], [ya, yb]])

    # Ensure we have at least 1 point
    if len(points) == 0:
        # Fallback to baseline
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        if in_unital(xa, xb, ya, yb) and not in_y_special(ya, yb):
                            continue
                        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
