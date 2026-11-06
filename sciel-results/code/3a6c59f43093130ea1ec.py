"""Fresh design: minimum-weight basis / shared-subexpression synthesis for XOR circuits.

Strategy:
- Represent each row (length m) as an int bitmask (bit i = column i).
- Target outputs = the m rows of the matrix (n rows total).
- Build a pool of candidate vectors: inputs (0..n-1), the zero vector, and the target rows.
- Greedily pick a small set S of "gate vectors" such that every target row is expressible
  as an XOR of <= 2 elements of S (a gate is an XOR of two pool vectors). This keeps each
  output at depth 2 (two XOR gates) while sharing intermediate vectors globally.
- The cost of a gate = number of times its vector is needed as an intermediate.
- Use randomized restarts with different tie-breaking / orderings to escape local minima.
- Deadline guard: track elapsed time and return the best (fewest total gates) result.
"""
import time
import random
import sys

def _row_to_int(row):
    v = 0
    for i, b in enumerate(row):
        if b:
            v |= (1 << i)
    return v

def _int_weight(x):
    return bin(x).count('1')

def synthesize(matrix, depth_limit):
    t0 = time.time()
    DEADLINE = 80.0  # safety margin under 90s kill
    # Parse matrix
    m = len(matrix[0])
    n = len(matrix)
    targets = [_row_to_int(row) for row in matrix]
    zero = 0

    # Baseline: each output = XOR of inputs directly.
    # Count gates needed if we don't share: sum over outputs of (popcount - 1)
    baseline_gates = 0
    for tv in targets:
        w = _int_weight(tv)
        if w > 1:
            baseline_gates += (w - 1)
    if baseline_gates == 0:
        # All rows have weight <= 1 -> trivial
        return {'gates': [], 'outputs': list(range(n))}

    # Wire naming: inputs 0..n-1, zero = n, gates n+1+k
    # We'll build a circuit: list of (a,b) gates producing wires n+1, n+2, ...
    # and outputs list of wire indices.

    # We need a set S of vectors such that each target = s1 ^ s2 for s1,s2 in S
    # (allowing s1=s2 gives 0; allow s1 or s2 to be an input or a gate vector).
    # S includes: all input unit vectors, zero, and some gate vectors.
    # A gate vector v is produced by an XOR gate (a,b) where a,b are wires whose
    # values are in S (inputs or earlier gate vectors). To keep depth small, we
    # produce all gate vectors from inputs directly (depth 1), so each gate vector
    # = input ^ input (2 inputs) -> depth 1. Then output = gate ^ input or gate^gate -> depth 2.

    # Simpler formulation: choose a multiset of gate vectors G (each produced by one
    # XOR gate from two inputs). Each target must equal (g ^ i) or (i1 ^ i2) [direct]
    # or (g1 ^ g2) or (g ^ 0)=g. To minimize gates, we want to cover all targets with
    # as few distinct gate vectors as possible, where a target is covered if it equals
    # g ^ u for some input unit u, or g ^ g2 for two gates, or direct input XOR.

    # Let's use a greedy set-cover: candidate gate vectors are all pairwise input XORs
    # that are "useful". For each target, the set of gate vectors that can express it
    # with one more XOR: target = g ^ u  =>  g = target ^ u. So for each target and each
    # input unit u, g = target ^ u is a candidate gate. Also target could be g itself
    # (if target has weight 2, g=target, expressed as g ^ 0). And target = g1^g2.

    # To keep it tractable, restrict to: each target = g ^ u (one gate + one input) OR
    # target = g (gate only, weight-2 targets) OR target = u1 ^ u2 (direct, weight 2).
    # This covers all targets with depth 2 and uses gate vectors of form target^u.

    # Build candidate gate pool
    candidates = set()
    for tv in targets:
        # g = tv ^ u for each unit u
        for i in range(m):
            u = 1 << i
            candidates.add(tv ^ u)
        # g = tv (if we allow output = gate directly, i.e., gate ^ zero)
        candidates.add(tv)
    candidates = [c for c in candidates if c != 0]

    # For each target, which candidates cover it (with one extra XOR to an input or zero)
    target_cover = []
    for tv in targets:
        cov = []
        for gi, c in enumerate(candidates):
            # tv = c ^ u  => u = tv ^ c must be a single unit or 0
            u = tv ^ c
            if u == 0:
                cov.append((gi, -1))  # output = gate directly
            elif (u & (u - 1)) == 0:  # single bit
                cov.append((gi, u.bit_length() - 1))
        target_cover.append(cov)

    # Greedy set cover with randomized restarts
    best_gates = None
    best_count = None

    rng = random.Random(12345)

    def try_greedy(order_seed):
        local_rng = random.Random(order_seed)
        # weights: how many targets each candidate covers
        cover_count = [0] * len(candidates)
        for cov in target_cover:
            seen = set()
            for gi, _ in cov:
                if gi not in seen:
                    cover_count[gi] += 1
                    seen.add(gi)

        chosen = set()
        remaining = list(range(n))
        # Precompute for each target the list of (gi) that cover it
        # Iterate: pick candidate covering most remaining targets
        # To speed, maintain for each target its available candidates
        avail = [set() for _ in range(n)]
        for t in range(n):
            for gi, _ in target_cover[t]:
                avail[t].add(gi)

        # Map candidate -> set of targets it covers
        cand_targets = [set() for _ in range(len(candidates))]
        for t in range(n):
            for gi in avail[t]:
                cand_targets[gi].add(t)

        while remaining:
            if time.time() - t0 > DEADLINE:
                return None
            # pick candidate with max coverage among remaining, with random tie-break
            best_gi = None
            best_cov = -1
            best_rand = None
            for gi in range(len(candidates)):
                if gi in chosen:
                    continue
                cnt = len(cand_targets[gi] & set(remaining))
                if cnt <= 0:
                    continue
                r = local_rng.random()
                if best_gi is None or cnt > best_cov or (cnt == best_cov and r > best_rand):
                    best_gi = gi
                    best_cov = cnt
                    best_rand = r
            if best_gi is None:
                # cannot cover; fallback: use direct input XOR for remaining
                return None
            chosen.add(best_gi)
            # remove covered targets
            newly = cand_targets[best_gi] & set(remaining)
            remaining = [t for t in remaining if t not in newly]
            # update cand_targets for other candidates (remove newly covered)
            # (lazy: we recompute intersection with remaining each time)

        return chosen

    # Try several random orderings
    for trial in range(200):
        if time.time() - t0 > DEADLINE:
            break
        chosen = try_greedy(trial * 7919 + 1)
        if chosen is None:
            continue
        cnt = len(chosen)
        if best_count is None or cnt < best_count:
            best_count = cnt
            best_gates = chosen
        if best_count <= max(1, baseline_gates // 3):
            break  # good enough

    if best_gates is None:
        # Fallback: no sharing, direct synthesis
        gates = []
        outputs = []
        next_wire = n + 1
        for tv in targets:
            bits = [i for i in range(m) if (tv >> i) & 1]
            if len(bits) <= 1:
                outputs.append(bits[0] if bits else n)  # zero wire if empty
            else:
                acc = bits[0]
                for b in bits[1:]:
                    gates.append([acc, b])
                    acc = next_wire
                    next_wire += 1
                outputs.append(acc)
        return {'gates': gates, 'outputs': outputs}

    # Build the circuit from chosen gate vectors
    # Each chosen gate vector c must be produced by an XOR of two inputs (since c = tv ^ u,
    # and tv has some bits; c may have weight > 2, so we may need to produce c itself
    # via multiple XORs!). Wait: c = tv ^ u could have high weight. To produce c as a
    # single gate, we need c to be XOR of two existing wires. But inputs are unit vectors,
    # so a single gate from two inputs gives weight <= 2. So c with weight > 2 can't be
    # one gate from two inputs.

    # Reconsider: a "gate" is an XOR of two wires. To produce a vector c of weight w,
    # we need w-1 gates (chain of XORs of inputs). So the cost of gate vector c is
    # weight(c) - 1 gates. Then output = c ^ u uses 1 more gate (or 0 if u part is zero).
    # So total gates = sum over chosen c of (weight(c)-1) + (number of outputs needing
    # an extra gate) . Let me recompute cost properly.

    # Recompute: for each chosen candidate c, producing c costs (weight(c)-1) gates
    # (if weight(c)>=2; if weight(c)==1 it's an input, cost 0; weight 0 cost 0).
    # For each target tv, we pick a covering candidate c and a unit u (or zero) such that
    # tv = c ^ u. If u is a unit input, output gate = c_wire ^ u_wire (1 gate). If u=0,
    # output = c_wire (0 extra gates).
    # Total gates = sum_{c in chosen} (max(0, weight(c)-1)) + (number of targets where u is a unit)

    # Let's recompute the best assignment given chosen set, and total cost.
    def total_cost(chosen_set):
        prod_cost = 0
        for gi in chosen_set:
            c = candidates[gi]
            w = _int_weight(c)
            if w >= 2:
                prod_cost += (w - 1)
        extra = 0
        for t in range(n):
            tv = targets[t]
            # find a chosen gi covering tv with minimal extra (prefer u=0)
            found_zero = False
            found_unit = False
            for gi, u_idx in target_cover[t]:
                if gi in chosen_set:
                    if u_idx == -1:
                        found_zero = True
                        break
                    else:
                        found_unit = True
            if found_zero:
                pass
            elif found_unit:
                extra += 1
            else:
                # not covered by chosen -> fallback direct
                w = _int_weight(tv)
                if w >= 2:
                    prod_cost += (w - 1)
        return prod_cost + extra

    # Recompute best with proper cost
    best_cost = None
    best_chosen = None
    for trial in range(200):
        if time.time() - t0 > DEADLINE:
            break
        chosen = try_greedy(trial * 7919 + 1)
        if chosen is None:
            continue
        c = total_cost(chosen)
        if best_cost is None or c < best_cost:
            best_cost = c
            best_chosen = chosen

    if best_chosen is None or best_cost >= baseline_gates:
        # Fallback direct
        gates = []
        outputs = []
        next_wire = n + 1
        for tv in targets:
            bits = [i for i in range(m) if (tv >> i) & 1]
            if len(bits) <= 1:
                outputs.append(bits[0] if bits else n)
            else:
                acc = bits[0]
                for b in bits[1:]:
                    gates.append([acc, b])
                    acc = next_wire
                    next_wire += 1
                outputs.append(acc)
        return {'gates': gates, 'outputs': outputs}

    # Build circuit for best_chosen
    gates = []
    next_wire = n + 1
    wire_of = {}  # vector int -> wire
    for i in range(n):
        wire_of[1 << i] = i  # input unit vectors (rows are length m; inputs are wires 0..n-1)
    wire_of[0] = n

    def get_wire(vec):
        if vec in wire_of:
            return wire_of[vec]
        # produce vec by chaining inputs
        bits = [i for i in range(m) if (vec >> i) & 1]
        if not bits:
            return n
        acc = bits[0]
        for b in bits[1:]:
            gates.append([acc, b])
            acc = next_wire
            next_wire += 1
        wire_of[vec] = acc
        return acc

    for gi in sorted(best_chosen):
        c = candidates[gi]
        if c != 0:
            get_wire(c)

    # Assign each target to a covering candidate
    outputs = []
    for t in range(n):
        tv = targets[t]
        chosen_pair = None
        for gi, u_idx in target_cover[t]:
            if gi in best_chosen:
                chosen_pair = (gi, u_idx)
                if u_idx == -1:
                    break
        if chosen_pair is None:
            # direct
            bits = [i for i in range(m) if (tv >> i) & 1]
            if not bits:
                outputs.append(n)
                continue
            if len(bits) == 1:
                outputs.append(bits[0])
                continue
            acc = bits[0]
            for b in bits[1:]:
                gates.append([acc, b])
                acc = next_wire
                next_wire += 1
            outputs.append(acc)
            continue
        gi, u_idx = chosen_pair
        c = candidates[gi]
        cw = get_wire(c)
        if u_idx == -1:
            outputs.append(cw)
        else:
            gates.append([cw, u_idx])
            outputs.append(next_wire)
            next_wire += 1

    return {'gates': gates, 'outputs': outputs}
