import time
import random
import numpy as np

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = P['n']
    m = P['m']
    N = 1 << n  # 4096

    # Precompute comparability matrix: comp[a][b] = 1 if a and b are comparable
    # comparable means a ⊆ b or b ⊆ a
    comp = np.zeros((N, N), dtype=np.int8)
    for a in range(N):
        for b in range(N):
            if (a & b) == a or (a & b) == b:
                comp[a][b] = 1
    # For counting pairs, we want strict: a ⊂ b or b ⊂ a (not equal)
    # But since all masks in family are distinct, a==b won't happen in pairs
    # However, for comp_in_F[y] we count how many family members are comparable to y
    # If y is in family, we need to be careful. Let's use comp as is (includes self-comparability)

    # For faster access, precompute for each mask a list of comparable masks
    # Actually, let's just use the matrix directly

    def count_comparable(family):
        """Count comparable pairs in family using precomputed matrix."""
        fam = np.array(family, dtype=np.int32)
        # Get the submatrix
        sub = comp[np.ix_(fam, fam)]
        # Upper triangle sum (excluding diagonal since distinct)
        count = int(np.triu(sub, 1).sum())
        return count

    def build_comp_in_F(family):
        """For each mask y, count how many family members are comparable to y."""
        # comp[family[i]][y] for all i, summed
        # Shape: (m, N) -> sum over axis 0 -> (N,)
        fam_arr = np.array(family, dtype=np.int32)
        comp_in_F = comp[fam_arr, :].sum(axis=0)
        return comp_in_F  # numpy array of shape (N,)

    def local_search(family, max_iters, deadline):
        """Local search with single-move swaps. Returns (best_family, best_score)."""
        current_score = count_comparable(family)
        best_score = current_score
        best_family = family[:]
        fam_set = set(family)
        comp_in_F = build_comp_in_F(family)  # numpy array

        for iteration in range(max_iters):
            if time.time() > deadline - 3:
                break

            improved = False
            for i in range(m):
                x = family[i]
                # k_x = number of other family members comparable to x
                # comp_in_F[x] includes x itself (since x is comparable to x)
                # So k_x = comp_in_F[x] - 1
                k_x = int(comp_in_F[x]) - 1

                best_y = -1
                best_val = -1

                # For each candidate y not in family:
                # new_comparables = comp_in_F[y] - (1 if y comparable to x else 0)
                # because removing x from family reduces comp_in_F by 1 for all y comparable to x
                # But we're adding y, so the net change is:
                # remove x: all y lose 1 if comparable to x
                # add y: y gains comp_in_F[y] - 1 (excluding itself)
                # Actually let me think more carefully.
                #
                # Current state: family F, score S
                # Remove x from F: new family F' = F \ {x}
                # For any y not in F:
                #   comp_in_F'[y] = comp_in_F[y] - (1 if y comparable to x else 0)
                #   where "comparable to x" means x⊆y or y⊆x
                # Add y to F': new family F'' = F' ∪ {y}
                # The new comparables for y in F'' is comp_in_F'[y] (since y not in F')
                # But we also need to count pairs between y and all elements in F'
                # That's exactly comp_in_F'[y]
                #
                # Change in score:
                # Old pairs involving x: k_x
                # New pairs involving y: comp_in_F'[y]
                # Delta = comp_in_F'[y] - k_x
                #
                # comp_in_F'[y] = comp_in_F[y] - (1 if comp[x][y] else 0)
                # Note: comp[x][y] is 1 if x and y are comparable (including if x==y, but y!=x here)

                # Vectorized: for all y not in fam_set
                # comp_in_F[y] - (comp[x,y] if y not in fam_set)
                # We can compute this for all y and mask out fam_set

                # comp[x, :] gives comparability of x with all masks
                # For y in fam_set (except x), we don't consider
                # For y not in fam_set: val = comp_in_F[y] - comp[x, y]
                #   (since if y is comparable to x, removing x reduces comp_in_F[y] by 1)

                # Build a boolean mask for candidates
                # This is expensive to do every iteration. Let's precompute.

                # Actually, let's just loop over N=4096 which is fast enough
                for y in range(N):
                    if y in fam_set:
                        continue
                    val = int(comp_in_F[y])
                    if comp[x, y]:
                        val -= 1
                    if val > best_val:
                        best_val = val
                        best_y = y

                if best_y == -1 or best_val <= k_x:
                    continue

                # Perform the swap: remove x, add best_y
                # Update comp_in_F:
                # Remove x: for all y, if comp[x,y] then comp_in_F[y] -= 1
                # Add best_y: for all y, if comp[best_y, y] then comp_in_F[y] += 1
                # Note: comp_in_F[x] will be updated but x is no longer in family
                # comp_in_F[best_y] will be updated

                comp_in_F -= comp[x, :]
                comp_in_F += comp[best_y, :]

                # But wait: comp[x, x] = 1, so comp_in_F[x] -= 1
                # comp[best_y, best_y] = 1, so comp_in_F[best_y] += 1
                # This is correct because after the swap:
                #   x is not in family, so comp_in_F[x] should reflect only other family members
                #   best_y is in family, so comp_in_F[best_y] includes self-comparability

                fam_set.remove(x)
                fam_set.add(best_y)
                family[i] = best_y
                current_score += best_val - k_x

                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]

                improved = True

            if not improved:
                break

        return best_family, best_score

    best_overall = None
    best_score_overall = 0

    # Multiple restarts with different random seeds
    for restart in range(30):
        if time.time() > deadline - 10:
            break

        random.seed(42 + restart)
        family = random.sample(range(N), m)

        fam, score = local_search(family, 150, deadline)

        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam

    # Also try a structured initial family: a chain plus random extras
    if time.time() < deadline - 15:
        chain = [0]
        cur = 0
        for b in range(n):
            cur |= (1 << b)
            chain.append(cur)
        remaining = [x for x in range(N) if x not in chain]
        random.seed(999)
        extra = random.sample(remaining, m - len(chain))
        family = chain + extra
        fam, score = local_search(family, 150, deadline)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam

    # Another structured start: all sets of size 5,6,7 (middle layers)
    if time.time() < deadline - 15:
        from itertools import combinations
        mid_sets = []
        for size in [4, 5, 6, 7]:
            for comb in combinations(range(n), size):
                mask = 0
                for b in comb:
                    mask |= (1 << b)
                mid_sets.append(mask)
        random.seed(1234)
        if len(mid_sets) >= m:
            family = random.sample(mid_sets, m)
            fam, score = local_search(family, 150, deadline)
            if score > best_score_overall:
                best_score_overall = score
                best_overall = fam

    # Final polish with more iterations
    if best_overall is not None and time.time() < deadline - 5:
        best_overall, best_score_overall = local_search(best_overall, 300, deadline)

    if best_overall is None:
        best_overall = list(range(m))

    result = []
    for mask in best_overall:
        result.append([b for b in range(n) if (mask >> b) & 1])

    return result
