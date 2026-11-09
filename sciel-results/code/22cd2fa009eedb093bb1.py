# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 600.0  # safety margin before 630s kill

p = 7
K = 3  # alpha^2 = 3
N_TOTAL = p**4  # 2401


def _point_to_idx(xa, xb, ya, yb):
    return ((xa * p + xb) * p * p + (ya * p + yb))


def _idx_to_point(idx):
    yb = idx % p
    ya = (idx // p) % p
    xb = (idx // (p * p)) % p
    xa = (idx // (p * p * p)) % p
    return xa, xb, ya, yb


def _generate_seed_excluded():
    """Return boolean array where True = point is in S (excluded from N)."""
    squares = set((u * u) % p for u in range(p))
    excluded = np.zeros(N_TOTAL, dtype=np.int8)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Unital condition: 2*xa + ya^2 - yb^2*K = 0 mod 7
                    if (2 * xa + ya * ya - yb * yb * K) % p == 0:
                        # y is special if ya is a square mod 7
                        if ya not in squares:
                            excluded[_point_to_idx(xa, xb, ya, yb)] = 1
    return excluded


def _precompute_lines():
    """
    Precompute the 2450 lines.
    Returns a list of arrays, each of length 49, containing the indices of points on that line.
    """
    lines = []
    # Direction 0: v = (0, 1), vertical
    # Lines parameterized by (xa, xb)
    for xa in range(p):
        for xb in range(p):
            line_pts = []
            for ya in range(p):
                for yb in range(p):
                    line_pts.append(_point_to_idx(xa, xb, ya, yb))
            lines.append(np.array(line_pts, dtype=np.int32))

    # Directions 1..49: v = (1, m) for m = ma + mb*alpha
    for ma in range(p):
        for mb in range(p):
            # Lines parameterized by (c_a, c_b)
            for ca in range(p):
                for cb in range(p):
                    line_pts = []
                    for xa in range(p):
                        for xb in range(p):
                            ya = (ca + ma * xa + 3 * mb * xb) % p
                            yb = (cb + ma * xb + mb * xa) % p
                            line_pts.append(_point_to_idx(xa, xb, ya, yb))
                    lines.append(np.array(line_pts, dtype=np.int32))

    return lines


def nikodym_set() -> np.ndarray:
    start = time.time()

    # Generate initial S from seed
    in_S = _generate_seed_excluded()

    # Precompute lines
    lines = _precompute_lines()
    n_lines = len(lines)

    # Precompute point_lines: for each point, list of line indices containing it
    point_lines = [[] for _ in range(N_TOTAL)]
    for i, line in enumerate(lines):
        for idx in line:
            point_lines[idx].append(i)

    # Compute initial bad counts
    # bad[x] = count of lines L containing x such that (count_S(L) - (1 if x in S else 0)) > 0
    line_s_counts = np.zeros(n_lines, dtype=np.int32)
    for i, line in enumerate(lines):
        line_s_counts[i] = np.sum(in_S[line])

    bad = np.zeros(N_TOTAL, dtype=np.int32)
    for i, line in enumerate(lines):
        cnt = line_s_counts[i]
        if cnt > 0:
            for idx in line:
                if cnt > (1 if in_S[idx] else 0):
                    bad[idx] += 1

    # Greedy: try to add points to S (i.e., remove from N) to maintain validity.
    # Wait - we want to MAXIMIZE N, so we want to MINIMIZE S.
    # The Nikodym condition: for every point x, there exists a line L through x
    # (in the allowed directions) such that all points of L \ {x} are in N.
    # Equivalently: for every point x, there exists a line L through x such that
    # L \ {x} is disjoint from S (i.e., no S points on L \ {x}).
    # 
    # A point x is "covered" (valid in N) iff there exists a line L through x
    # (with allowed direction) such that S ∩ (L \ {x}) = ∅.
    # Equivalently: bad[x] < 50, where bad[x] = number of directions where
    # S ∩ (L \ {x}) ≠ ∅.
    # 
    # So N must contain only points x with bad[x] < 50.
    # And S must be such that for all x in N, bad[x] < 50.
    # 
    # To maximize |N| = 2401 - |S|, we want |S| as small as possible.
    # But S must be "valid": removing any point from S would cause some x in N
    # to have bad[x] >= 50.
    # 
    # Actually, the constraint is just: for all x in N, bad[x] < 50.
    # We can start with a valid S and try to remove points from S (adding to N)
    # as long as validity is maintained.
    # 
    # Let's redefine: we want to find the smallest valid S.
    # Start with the seed S, then try to remove points.

    # Actually, let me re-read the problem.
    # "every point x of F_49^2 (in N or not) must lie on a line ... whose points other than x all belong to N"
    # So for EVERY x (whether in N or S), there must be a line L through x such that L \ {x} ⊆ N.
    # This means: for every x, there exists a line L through x (allowed direction) such that
    # S ∩ (L \ {x}) = ∅.
    # 
    # So the condition is on ALL points, not just those in N.
    # bad[x] = number of allowed-direction lines through x that have at least one S point (other than x itself if x in S).
    # Wait: if x in S, then L \ {x} must be disjoint from S. So S ∩ (L \ {x}) = ∅.
    # If x not in S, then L \ {x} must be disjoint from S. Same condition.
    # So in both cases: there exists a line L through x (allowed direction) such that
    # no other point of L is in S.
    # 
    # bad[x] = count of allowed-direction lines L through x such that (S ∩ L) \ {x} ≠ ∅.
    # Wait, more precisely: S ∩ (L \ {x}) ≠ ∅, i.e., there exists y in L, y ≠ x, with y in S.
    # 
    # x is valid iff bad[x] < 50 (there's at least one line through x with no other S points).
    # 
    # So the constraint is: for ALL x (all 2401 points), bad[x] < 50.
    # And N = complement of S.
    # We want to minimize |S| subject to: for all x, bad[x] < 50.
    # 
    # The seed gives a valid S. We want to shrink it.

    # Strategy: start with seed S, try to remove points one by one.
    # A point c in S can be removed if, after removal, for all x, bad_new[x] < 50.
    # Removing c from S:
    #   For each line L containing c: line_s_counts[L] decreases by 1.
    #   For each x on such L: bad[x] may decrease by 1.
    #   Specifically, bad[x] decreases by 1 if:
    #     OLD: S ∩ (L \ {x}) ≠ ∅
    #     NEW: S ∩ (L \ {x}) = ∅
    #   This happens when OLD count_S(L \ {x}) = 1 and that point is c.
    #   I.e., OLD: (line_s_counts[L] - (1 if x in S else 0)) = 1 and c is the only other S point.
    #   
    #   More carefully:
    #   OLD condition for bad increment on line L at point x:
    #     (line_s_counts_old[L] - (1 if in_S_old[x] else 0)) > 0
    #   NEW condition:
    #     (line_s_counts_new[L] - (1 if in_S_new[x] else 0)) > 0
    #   
    #   If x ≠ c: in_S_new[x] = in_S_old[x].
    #     line_s_counts_new[L] = line_s_counts_old[L] - 1.
    #     OLD: cnt_old - delta_x > 0
    #     NEW: cnt_old - 1 - delta_x > 0
    #     bad[x] decreases by 1 iff: (cnt_old - delta_x > 0) and (cnt_old - 1 - delta_x <= 0)
    #     i.e., cnt_old - delta_x == 1.
    #   
    #   If x = c: in_S_old[c] = 1, in_S_new[c] = 0.
    #     OLD: cnt_old - 1 > 0, i.e., cnt_old >= 2.
    #     NEW: (cnt_old - 1) - 0 > 0, i.e., cnt_old >= 2.
    #     Same condition! So bad[c] does NOT change when c is removed from S.
    #     Wait, that can't be right. Let me re-examine.
    #     If x = c, x was in S. L \ {x} = L \ {c}.
    #     OLD: S ∩ (L \ {c}) = (S \ {c}) ∩ (L \ {c}) = S_old ∩ (L \ {c}) since c in S.
    #       Actually S_old contains c. S_old ∩ (L \ {c}) = (S_old \ {c}) ∩ (L \ {c}).
    #       line_s_counts_old[L] = |S_old ∩ L|.
    #       |S_old ∩ (L \ {c})| = |S_old ∩ L| - 1 (since c in S_old ∩ L) = cnt_old - 1.
    #       OLD bad condition: cnt_old - 1 > 0, i.e., cnt_old >= 2.
    #     NEW: S_new = S_old \ {c}. S_new ∩ (L \ {c}) = S_new ∩ L = (S_old \ {c}) ∩ L.
    #       |S_new ∩ L| = cnt_old - 1.
    #       |S_new ∩ (L \ {c})| = cnt_old - 1 (since c not in S_new).
    #       NEW bad condition: (cnt_old - 1) - 0 > 0, i.e., cnt_old >= 2.
    #     So indeed, bad[c] is unchanged. Makes sense: removing c from S doesn't change
    #     whether there are other S points on L \ {c}.
    
    # So to check if we can remove c from S:
    #   For each line L containing c:
    #     cnt = line_s_counts[L]
    #     If cnt == 1: c is the only S point on L.
    #       For x on L, x ≠ c: OLD delta: cnt - (1 if in_S[x] else 0) = 1 - (1 if in_S[x] else 0).
    #         If in_S[x] = 0: 1 - 0 = 1 > 0. So bad was incremented.
    #         If in_S[x] = 1: 1 - 1 = 0. Not incremented. But wait, if in_S[x] = 1 and x ≠ c,
    #         then cnt >= 2. Contradiction with cnt == 1. So all x ≠ c on L have in_S[x] = 0.
    #         So for all x ≠ c on L: OLD bad condition: 1 - 0 = 1 > 0. TRUE.
    #         NEW: cnt_new = 0. 0 - 0 = 0 > 0. FALSE.
    #         So bad[x] decreases by 1 for all x ≠ c on L.
    #         This is GOOD (reduces bad counts).
    #     If cnt == 2: c and one other point s in S on L.
    #       For x on L, x ≠ c, x ≠ s: in_S[x] = 0.
    #         OLD: 2 - 0 = 2 > 0. TRUE.
    #         NEW: 1 - 0 = 1 > 0. TRUE. No change.
    #       For x = s: in_S[x] = 1.
    #         OLD: 2 - 1 = 1 > 0. TRUE.
    #         NEW: 1 - 1 = 0 > 0. FALSE.
    #         So bad[s] decreases by 1.
    #       For x = c: unchanged as shown.
    #     If cnt >= 3:
    #       For x on L, x ≠ c, x ≠ s (any S point): in_S[x] = 0.
    #         OLD: cnt > 0. NEW: cnt - 1 > 0 (since cnt >= 3, cnt-1 >= 2 > 0). No change.
    #       For x = s (another S point): in_S[x] = 1.
    #         OLD: cnt - 1 > 0 (cnt >= 3, so cnt-1 >= 2 > 0). TRUE.
    #         NEW: cnt - 1 - 1 = cnt - 2 > 0 (cnt >= 3, cnt-2 >= 1 > 0). TRUE. No change.
    #       So no bad counts change for cnt >= 3.
    
    # Therefore, removing c from S can only DECREASE bad counts (or leave them unchanged).
    # It can NEVER increase bad counts!
    # So if the current S is valid (all bad[x] < 50), removing any point c from S
    # keeps it valid or makes it more valid.
    # 
    # Wait, that means we should remove ALL points from S! Then S = ∅, N = all points.
    # Is that valid? If S = ∅, then for any x and any line L through x,
    # S ∩ (L \ {x}) = ∅. So bad[x] = 0 for all x. All x are valid.
    # So N = F_49^2 is a valid Nikodym set!
    # 
    # But wait, that seems too easy. Let me re-read the problem statement.
    # "every point x of F_49^2 (in N or not) must lie on a line {x + t*v : t in F_49}, 
    #  with v = (1, m) for some m in F_49 or v = (0, 1), whose points other than x all belong to N."
    # 
    # If N = F_49^2, then for any x, pick any allowed line L through x. 
    # All points of L \ {x} are in N (since N is everything). So the condition is satisfied.
    # 
    # So the answer is just all 2401 points?!
    # 
    # Hmm, but the seed program returns a proper subset. And the score is based on 
    # complement size 2401 - |N|. If N = all points, complement = 0, score = 0.
    # Wait, "Metric (higher is better): the complement size 2401 - |N|".
    # So we want to MAXIMIZE the complement, i.e., MINIMIZE |N|?
    # No wait: "the score is proportional to it" where "it" is the complement size.
    # So higher complement = higher score. We want N to be as SMALL as possible.
    # 
    # But N must be a Nikodym set. And the Nikodym condition must hold for ALL points.
    # So we need the smallest valid N (equivalently, the largest valid S).
    # 
    # OK so I had it backwards. Let me restart.
    # 
    # We want to MAXIMIZE |S| (minimize |N|) such that for all x, bad[x] < 50.
    # 
    # Start with seed S. Try to ADD points to S.
    # A point c not in S can be added if, after adding, for all x, bad_new[x] < 50.
    # 
    # Adding c to S:
    #   For each line L containing c: line_s_counts[L] increases by 1.
    #   bad[x] may increase by 1 for some x on L.
    #   Specifically, bad[x] increases by 1 if:
    #     OLD: (cnt - (1 if in_S[x] else 0)) <= 0
    #     NEW: (cnt + 1 - (1 if in_S_new[x] else 0)) > 0
    #   
    #   If x ≠ c: in_S_new[x] = in_S[x].
    #     OLD: cnt - delta <= 0.
    #     NEW: cnt + 1 - delta > 0.
    #     This happens iff cnt - delta == 0, i.e., cnt == delta.
    #     If delta = 0 (x not in S): cnt == 0.
    #     If delta = 1 (x in S): cnt == 1.
    #   
    #   If x = c: in_S_old[c] = 0, in_S_new[c] = 1.
    #     OLD: cnt - 0 <= 0, i.e., cnt == 0.
    #     NEW: cnt + 1 - 1 = cnt > 0.
    #     So if cnt == 0: OLD false (0 <= 0 is true, wait).
    #     Let me redo. OLD condition for bad: (cnt - (1 if in_S_old[x] else 0)) > 0.
    #     For x = c: in_S_old[c] = 0. OLD: cnt > 0.
    #     NEW: in_S_new[c] = 1. NEW: (cnt + 1) - 1 = cnt > 0.
    #     Same condition! So bad[c] doesn't change.
    #     Wait, that's the same as before. Let me verify.
    #     OLD: S ∩ (L \ {c}). c not in S. So S ∩ (L \ {c}) = S ∩ L.
    #       |S ∩ L| = cnt. bad condition: cnt > 0.
    #     NEW: S_new = S ∪ {c}. S_new ∩ (L \ {c}) = (S ∪ {c}) ∩ (L \ {c}) = S ∩ (L \ {c}) since c notin L\{c}.
    #       |S ∩ (L \ {c})| = |S ∩ L| (since c notin S) = cnt.
    #       bad condition: cnt > 0.
    #     Same. OK.
    
    # So adding c to S:
    #   For each line L containing c with cnt = line_s_counts[L]:
    #     If cnt == 0: bad[x] increases by 1 for all x on L, x ≠ c, x not in S.
    #     If cnt == 1: bad[x] increases by 1 for all x on L, x ≠ c, x in S.
    #     If cnt >= 2: no change.
    #   And bad[c] is unchanged.
    
    # To add c: need to check that for all affected x, bad[x] + 1 < 50, i.e., bad[x] < 49.
    
    # This is what the original code does. Let me just run it properly.

    # Greedy: try to add points to S.
    # A point c not in S can be added if for all lines L containing c,
    # the points that would have bad incremented all have bad < 49.

    # Candidates: all points not in S, in random order.
    candidates = [i for i in range(N_TOTAL) if not in_S[i]]
    np.random.seed(42)
    np.random.shuffle(candidates)

    added = 0
    for c_idx in candidates:
        if time.time() > _DEADLINE - 30:
            break

        can_add = True

        for l_idx in point_lines[c_idx]:
            line = lines[l_idx]
            cnt = line_s_counts[l_idx]

            if cnt == 0:
                # Check points in line not in S (x ≠ c_idx)
                for x in line:
                    if x != c_idx and not in_S[x]:
                        if bad[x] >= 49:
                            can_add = False
                            break
                if not can_add:
                    break
            elif cnt == 1:
                # Check points in line in S (x ≠ c_idx)
                for x in line:
                    if in_S[x]:  # x != c_idx guaranteed since c_idx not in S
                        if bad[x] >= 49:
                            can_add = False
                            break
                if not can_add:
                    break
            # cnt >= 2: no change

        if can_add:
            in_S[c_idx] = 1
            added += 1

            for l_idx in point_lines[c_idx]:
                line = lines[l_idx]
                cnt = line_s_counts[l_idx]
                line_s_counts[l_idx] += 1

                if cnt == 0:
                    for x in line:
                        if x != c_idx and not in_S[x]:
                            bad[x] += 1
                elif cnt == 1:
                    for x in line:
                        if in_S[x]:
                            bad[x] += 1
                            break

            if time.time() > _DEADLINE - 60:
                break

    # Build result: N = complement of S
    points = []
    for idx in range(N_TOTAL):
        if not in_S[idx]:
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])

    if len(points) == 0:
        # Fallback: return all points
        for idx in range(N_TOTAL):
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
