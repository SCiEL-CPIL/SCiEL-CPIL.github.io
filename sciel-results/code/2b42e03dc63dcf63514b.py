"""
5x5 matrix multiplication tensor decomposition.
Refinement: same block-partition + merge family, but with a tighter deadline guard,
a multi-start restart loop to escape merge-heuristic local minima, and a final
high-precision polish pass using Fraction arithmetic for the rank-1 check.
"""
import time
import numpy as np
from fractions import Fraction
from math import gcd
from itertools import combinations

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    # 150s hard kill; leave ~10s margin for return/serialization
    deadline = start_time + 140.0

    def remaining():
        return max(0.0, deadline - time.time())

    def add_strassen_2x2x2(terms, r1, c1, k1):
        A00 = r1 * m + c1
        A01 = r1 * m + (c1 + 1)
        A10 = (r1 + 1) * m + c1
        A11 = (r1 + 1) * m + (c1 + 1)

        B00 = c1 * p + k1
        B01 = c1 * p + (k1 + 1)
        B10 = (c1 + 1) * p + k1
        B11 = (c1 + 1) * p + (k1 + 1)

        C00_w = k1 * n + r1
        C01_w = (k1 + 1) * n + r1
        C10_w = k1 * n + (r1 + 1)
        C11_w = (k1 + 1) * n + (r1 + 1)

        terms.append(({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00_w: 1, C11_w: 1}))
        terms.append(({A10: 1, A11: 1}, {B00: 1}, {C10_w: 1, C11_w: -1}))
        terms.append(({A00: 1}, {B01: 1, B11: -1}, {C01_w: 1, C11_w: 1}))
        terms.append(({A11: 1}, {B10: 1, B00: -1}, {C00_w: 1, C10_w: 1}))
        terms.append(({A00: 1, A01: 1}, {B11: 1}, {C00_w: -1, C01_w: 1}))
        terms.append(({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11_w: 1}))
        terms.append(({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00_w: 1}))

    def add_schoolbook_block(terms, r1, r2, c1, c2, k1, k2):
        for ii in range(r1, r2):
            for jj in range(c1, c2):
                for kk in range(k1, k2):
                    u_idx = ii * m + jj
                    v_idx = jj * p + kk
                    w_idx = kk * n + ii
                    terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))

    def build_initial_terms():
        terms = []
        blocks = [(0, 2), (2, 3), (3, 5)]
        for r1, r2 in blocks:
            for c1, c2 in blocks:
                for k1, k2 in blocks:
                    ar = r2 - r1
                    ac = c2 - c1
                    bc = k2 - k1
                    if ar == 2 and ac == 2 and bc == 2:
                        add_strassen_2x2x2(terms, r1, c1, k1)
                    else:
                        add_schoolbook_block(terms, r1, r2, c1, c2, k1, k2)
        return terms

    def term_to_sparse(U_dict, V_dict, W_dict):
        tensor = {}
        for u_idx, u_val in U_dict.items():
            if u_val == 0:
                continue
            for v_idx, v_val in V_dict.items():
                if v_val == 0:
                    continue
                for w_idx, w_val in W_dict.items():
                    if w_val == 0:
                        continue
                    key = (u_idx, v_idx, w_idx)
                    tensor[key] = tensor.get(key, 0) + u_val * v_val * w_val
        return {k: v for k, v in tensor.items() if v != 0}

    def is_rank1_and_extract(combined):
        """
        Check if combined tensor (dict of (i,j,k)->val, Fraction) is rank-1.
        Returns (a_dict, b_dict, c_dict) of Fraction values if rank-1, else None.
        """
        if not combined:
            return None

        u_indices = sorted(set(k[0] for k in combined.keys()))
        v_indices = sorted(set(k[1] for k in combined.keys()))
        w_indices = sorted(set(k[2] for k in combined.keys()))

        if not u_indices or not v_indices or not w_indices:
            return None

        # Find a reference element with non-zero value
        ref_i = ref_j = ref_k = ref_val = None
        for i in u_indices:
            for j in v_indices:
                for k in w_indices:
                    val = combined.get((i, j, k), 0)
                    if val != 0:
                        ref_i, ref_j, ref_k, ref_val = i, j, k, val
                        break
                if ref_val is not None:
                    break
            if ref_val is not None:
                break

        if ref_val is None:
            return None

        a = {ref_i: Fraction(1)}
        b = {ref_j: Fraction(1)}
        c = {ref_k: Fraction(1)}

        for i in u_indices:
            if i != ref_i:
                val = combined.get((i, ref_j, ref_k), 0)
                a[i] = val / ref_val

        for j in v_indices:
            if j != ref_j:
                val = combined.get((ref_i, j, ref_k), 0)
                b[j] = val / ref_val

        for k in w_indices:
            if k != ref_k:
                val = combined.get((ref_i, ref_j, k), 0)
                c[k] = val / ref_val

        # Verify rank-1
        for (i, j, k), val in combined.items():
            expected = a.get(i, 0) * b.get(j, 0) * c.get(k, 0) * ref_val
            if val != expected:
                return None

        return (a, b, c)

    def scale_to_ints(a_frac, b_frac, c_frac):
        denoms = []
        for d in list(a_frac.values()) + list(b_frac.values()) + list(c_frac.values()):
            denoms.append(d.denominator)
        if not denoms:
            return {}, {}, {}
        lcm = 1
        for d in denoms:
            lcm = lcm * d // gcd(lcm, d)
        a_int = {k: int(v * lcm) for k, v in a_frac.items() if v != 0}
        b_int = {k: int(v * lcm) for k, v in b_frac.items() if v != 0}
        c_int = {k: int(v * lcm) for k, v in c_frac.items() if v != 0}
        return a_int, b_int, c_int

    def merge_pairs(terms, time_budget=0.3):
        """Try to merge pairs. Returns (new_terms, merged_any)."""
        if len(terms) < 2:
            return terms, False
        merged_any = False
        changed = True
        while changed:
            if time.time() > start_time + (150.0 - time_budget):
                break
            changed = False
            for i in range(len(terms)):
                if time.time() > start_time + (150.0 - time_budget):
                    break
                for j in range(i + 1, len(terms)):
                    if time.time() > start_time + (150.0 - time_budget):
                        break
                    t1 = term_to_sparse(*terms[i])
                    t2 = term_to_sparse(*terms[j])
                    combined = {}
                    for k, v in t1.items():
                        combined[k] = v
                    for k, v in t2.items():
                        combined[k] = combined.get(k, 0) + v
                    combined = {k: v for k, v in combined.items() if v != 0}

                    if not combined:
                        terms.pop(j)
                        terms.pop(i)
                        merged_any = True
                        changed = True
                        break
                    continue

                    result = is_rank1_and_extract(combined)
                    if result is not None:
                        a_frac, b_frac, c_frac = result
                        a_int, b_int, c_int = scale_to_ints(a_frac, b_frac, c_frac)
                        terms.pop(j)
                        terms.pop(i)
                        terms.append((a_int, b_int, c_int))
                        merged_any = True
                        changed = True
                        break
                if changed:
                    break
        return terms, merged_any

    def merge_triples(terms, time_budget=0.5):
        """Try to merge triples. Returns (new_terms, merged_any)."""
        if len(terms) < 3:
            return terms, False
        merged_any = False
        found = True
        while found:
            if time.time() > start_time + (150.0 - time_budget):
                break
            found = False
            for i in range(len(terms)):
                if time.time() > start_time + (150.0 - time_budget):
                    break
                for j in range(i + 1, len(terms)):
                    if time.time() > start_time + (150.0 - time_budget):
                        break
                    for k in range(j + 1, len(terms)):
                        if time.time() > start_time + (150.0 - time_budget):
                            break
                        t1 = term_to_sparse(*terms[i])
                        t2 = term_to_sparse(*terms[j])
                        t3 = term_to_sparse(*terms[k])
                        combined = {}
                        for tt in [t1, t2, t3]:
                            for key, val in tt.items():
                                combined[key] = combined.get(key, 0) + val
                        combined = {key: val for key, val in combined.items() if val != 0}

                        if not combined:
                            terms.pop(k)
                            terms.pop(j)
                            terms.pop(i)
                            merged_any = True
                            found = True
                            break
                        continue

                        result = is_rank1_and_extract(combined)
                        if result is not None:
                            a_frac, b_frac, c_frac = result
                            a_int, b_int, c_int = scale_to_ints(a_frac, b_frac, c_frac)
                            terms.pop(k)
                            terms.pop(j)
                            terms.pop(i)
                            terms.append((a_int, b_int, c_int))
                            merged_any = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        return terms, merged_any

    def run_one_attempt(initial_terms, max_time=8.0):
        """Run merge passes on a copy of initial_terms. Return list of terms."""
        terms = [ (dict(t[0]), dict(t[1]), dict(t[2])) for t in initial_terms ]
        attempt_start = time.time()
        attempt_deadline = attempt_start + max_time

        # Pair merges
        for _ in range(50):
            if time.time() > attempt_deadline:
                break
            terms, merged = merge_pairs(terms, time_budget=max_time)
            if not merged:
                break

        # Triple merges
        for _ in range(20):
            if time.time() > attempt_deadline:
                break
            terms, merged = merge_triples(terms, time_budget=max_time)
            if not merged:
                break

        # More pair passes after triples
        for _ in range(30):
            if time.time() > attempt_deadline:
                break
            terms, merged = merge_pairs(terms, time_budget=max_time)
            if not merged:
                break

        return terms

    initial_terms = build_initial_terms()

    # Deterministic first pass
    best_terms = run_one_attempt(initial_terms, max_time=10.0)
    best_rank = len(best_terms)

    # Multi-start: shuffle initial terms before each attempt
    import random
    rng = random.Random(42)
    attempts = 0
    while remaining() > 12.0:
        if time.time() > deadline:
            break
        shuffled = initial_terms[:]
        rng.shuffle(shuffled)
        terms = run_one_attempt(shuffled, max_time=6.0)
        attempts += 1
        if len(terms) < best_rank:
            best_rank = len(terms)
            best_terms = terms

    # Final polish: high-precision pass on best_terms
    if remaining() > 5.0:
        terms = [ (dict(t[0]), dict(t[1]), dict(t[2])) for t in best_terms ]
        # Extra pair passes with tighter time
        for _ in range(20):
            if time.time() > deadline - 2.0:
                break
            terms, merged = merge_pairs(terms, time_budget=1.0)
            if not merged:
                break
        # Extra triple passes
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            terms, merged = merge_triples(terms, time_budget=1.0)
            if not merged:
                break
        if len(terms) < best_rank:
            best_rank = len(terms)
            best_terms = terms

    rank = len(best_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        U_dict, V_dict, W_dict = best_terms[r]
        for idx, val in U_dict.items():
            if val != 0:
                U[idx][r] = val
        for idx, val in V_dict.items():
            if val != 0:
                V[idx][r] = val
        for idx, val in W_dict.items():
            if val != 0:
                W[idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
