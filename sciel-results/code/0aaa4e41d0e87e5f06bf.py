import numpy as np
import time
import random
import sys

_p = 7
_kq = 3
_Q = 49
_SLOTS = 2401
_INF = 10  # vertical direction index

def _encode(xa, xb, ya, yb):
    return ((xa * 7 + xb) * 49) + (ya * 7 + yb)

def _xy_from_slot(s):
    yb = s % 7; s //= 7
    ya = s % 7; s //= 7
    xb = s % 7; s //= 7
    xa = s % 7
    return xa, xb, ya, yb

# Precompute all points
_PTS = np.zeros((_SLOTS, 4), dtype=np.uint8)
for s in range(_SLOTS):
    xa, xb, ya, yb = _xy_from_slot(s)
    _PTS[s] = [xa, xb, ya, yb]

# Precompute lines for each direction
_lines = []
_line_of = []

for d in range(50):
    if d == _INF:
        # v = (0,1): lines are vertical, grouped by x
        dl = []
        d_lof = np.zeros(_SLOTS, dtype=np.int16)
        for xa in range(7):
            for xb in range(7):
                line = []
                for ya in range(7):
                    for yb in range(7):
                        s = _encode(xa, xb, ya, yb)
                        line.append(s)
                        d_lof[s] = len(dl)
                dl.append(np.array(line, dtype=np.int32))
        _lines.append(dl)
        _line_of.append(d_lof)
    else:
        da, db = (d // 7, d % 7)
        # v = (1, da + db*alpha)
        dl = []
        d_lof = np.zeros(_SLOTS, dtype=np.int16)
        for ya0 in range(7):
            for yb0 in range(7):
                line = []
                for ta in range(7):
                    for tb in range(7):
                        # t = ta + tb*alpha
                        # x = 0 + t = t
                        # y = y0 + t*(da + db*alpha)
                        txa = (ta * da + tb * db * _kq) % _p
                        txb = (ta * db + tb * da) % _p
                        nxa = ta % _p
                        nxb = tb % _p
                        nya = (ya0 + txa) % _p
                        nyb = (yb0 + txb) % _p
                        s = _encode(nxa, nxb, nya, nyb)
                        line.append(s)
                        d_lof[s] = len(dl)
                dl.append(np.array(line, dtype=np.int32))
        _lines.append(dl)
        _line_of.append(d_lof)

def _compute_Scount(S):
    """Compute Scount[d][s] = number of S points on line through s in direction d (including s if in S)."""
    Scount = np.zeros((50, _SLOTS), dtype=np.int16)
    S_arr = np.frombuffer(S, dtype=np.uint8) if isinstance(S, bytes) else np.array(S, dtype=np.uint8)
    for d in range(50):
        for line in _lines[d]:
            c = np.sum(S_arr[line])
            Scount[d, line] = c
    return Scount

def _is_valid(S):
    """
    Check if S (complement of N) gives a valid Nikodym set.
    Condition: For every point x, there exists a direction d such that the line through x in direction d
    has no S-points other than x.
    i.e., Scount[d][x] - S[x] == 0.
    """
    S_arr = np.frombuffer(S, dtype=np.uint8) if isinstance(S, bytes) else np.array(S, dtype=np.uint8)
    Scount = _compute_Scount(S)
    for s in range(_SLOTS):
        ss = S_arr[s]
        ok = False
        for d in range(50):
            if Scount[d, s] - ss == 0:
                ok = True
                break
        if not ok:
            return False
    return True

def _to_array(S):
    """Convert S (bytearray or bytes) to the output array of N = complement of S."""
    S_arr = np.frombuffer(S, dtype=np.uint8) if isinstance(S, bytes) else np.array(S, dtype=np.uint8)
    pts = []
    for s in range(_SLOTS):
        if not S_arr[s]:
            xa, xb, ya, yb = _xy_from_slot(s)
            pts.append([xa, xb, ya, yb])
    return np.array(pts, dtype=np.int64).reshape(-1, 2, 2)

def _local_search(deadline):
    """
    Strategy: Start with a random S, then iteratively try to increase |S| by adding points 
    that don't break validity, or repair invalid S by removing points.
    
    Actually, a better heuristic for this specific constraint (Nikodym) is to observe that 
    if we pick S to be a "random" set with density p, the probability a point is "blocked" in all 50 directions 
    is roughly (1 - (1-p)^48)^50. For small p, this is small.
    
    Let's use a "Random Walk" on the space of S.
    State: S (bytearray of 0/1).
    Move: Flip a random bit.
    Objective: Maximize |S| subject to Validity.
    
    Since Validity is a hard constraint, we can't just flip randomly.
    Instead, we can use the following:
    1. Start with S = empty.
    2. Try to add points one by one. If adding p makes the set invalid, we can try to "fix" it by removing some other points.
    
    However, the previous greedy approach failed to reach >1.0. The inspiration says "Greedy set cover minimizes R, maximizing S size to ~300".
    Wait, the score is 2401 - |N| = |S|. So maximizing S is the goal.
    The current program gets score 0.9864, which implies |S| is roughly 0.9864 * (seed_size)? No, the score is proportional to complement size.
    The seed score is 1.0. If seed is empty S, score is 0? No, "1.0 equals the seed program".
    Let's assume the seed program produces a valid set with some size.
    
    Let's try a different angle: **Simulated Annealing** or **Random Restart Greedy with Backtracking**.
    
    Actually, let's look at the structure.
    A point x is satisfied if there is a direction d such that Line_d(x) \ {x} contains NO points of S.
    Let's call such a direction a "witness" for x.
    
    If we choose S to be a random subset with probability p, the expected size is 2401*p.
    The probability that x is NOT satisfied is the probability that for ALL d, Line_d(x) \ {x} intersects S.
    For a fixed d, the probability that Line_d(x) \ {x} does NOT intersect S is (1-p)^48.
    So probability it DOES intersect is 1 - (1-p)^48.
    Assuming independence (which is false, but a good approximation for small p),
    P(x not satisfied) ≈ (1 - (1-p)^48)^50.
    
    We want P(x not satisfied) to be very small, say < 1/2401, so that likely all points are satisfied.
    (1 - (1-p)^48)^50 < 1/2401
    1 - (1-p)^48 < (1/2401)^(1/50) ≈ 0.89
    (1-p)^48 > 0.11
    1-p > 0.11^(1/48) ≈ 0.953
    p < 0.047
    
    So if we pick p ~ 0.04, we expect |S| ≈ 2401 * 0.04 ≈ 96.
    This might be better than the greedy approach if the greedy one is stuck.
    
    Let's try to construct S randomly with p=0.05, then check validity.
    If invalid, remove points that are "causing" the most conflicts?
    Or simply: Generate many random candidates and keep the largest valid one.
    
    To speed up validity check, we can use the Scount method.
    
    Let's implement a "Random Sampling with Repair" strategy.
    1. Generate S_random with p.
    2. Check validity.
    3. If invalid, identify unsatisfied points.
    4. For each unsatisfied point x, all its lines have S-points.
       We need to remove S-points to make at least one line clean.
       Greedy repair: For an unsatisfied x, pick a direction d, and remove ALL S-points on Line_d(x) except x (if x in S).
       This is expensive.
       
    Alternative:
    Use the "Greedy Set Cover" inspiration.
    The problem is equivalent to finding a large set S such that no point is "covered" by S in all directions.
    
    Let's stick to a robust **Random Restart Greedy** but optimize the inner loop using NumPy vectorization where possible, or simply accept the Python speed if we limit restarts.
    The previous code did 10 restarts. Maybe we can do more if we make the check faster?
    
    Let's try **Local Search from a Random Start**.
    1. Start with S = random set with p=0.1.
    2. While time remains:
       a. Compute Scount.
       b. Find unsatisfied points U.
       c. If U is empty, we have a valid S. Try to grow it:
          - Pick a random point q not in S.
          - Check if adding q keeps it valid.
          - If yes, add q.
          - If no, skip.
       d. If U is not empty, we must repair.
          - Pick a point x in U.
          - Pick a direction d for x.
          - Remove all S-points on Line_d(x) \ {x}.
          - This reduces |S| but might satisfy x.
    
    This "Grow then Repair" might find larger sets.
    """
    best_S = bytearray(_SLOTS)
    best_size = 0
    rng = random.Random(12345)
    
    # Precompute line neighbors for faster access?
    # _lines[d][line_idx] is already an array.
    
    def _check_and_fix(S_arr, Scount):
        """
        Returns a list of unsatisfied points.
        """
        unsat = []
        for s in range(_SLOTS):
            ss = S_arr[s]
            ok = False
            for d in range(50):
                if Scount[d, s] - ss == 0:
                    ok = True
                    break
            if not ok:
                unsat.append(s)
        return unsat

    for restart in range(20): # More restarts
        if time.time() > deadline - 10: break
        
        # Initialize S
        S_arr = np.zeros(_SLOTS, dtype=np.uint8)
        # Random initialization with small p
        p_init = 0.05
        S_arr[rng.choice(_SLOTS, size=int(_SLOTS * p_init), replace=False)] = 1
        S_ba = S_arr.tobytes()
        
        # Compute initial Scount
        Scount = _compute_Scount(S_ba)
        unsat = _check_and_fix(S_ba, Scount)
        
        # If initially invalid, try to repair by removing points
        # This is a heuristic repair
        max_repairs = 1000
        repairs = 0
        while unsat and repairs < max_repairs:
            if time.time() > deadline - 10: break
            # Pick a random unsatisfied point
            x = unsat[rng.randrange(len(unsat))]
            # Pick a random direction
            d = rng.randrange(50)
            # Remove all S points on Line_d(x) except x
            line_idx = _line_of[d][x]
            line = _lines[d][line_idx]
            changed = False
            for q in line:
                if q != x and S_arr[q] == 1:
                    S_arr[q] = 0
                    changed = True
            if not changed:
                # This direction was already clean? That shouldn't happen if x is unsatisfied.
                # Unless x is in S and Scount includes x?
                # If x is in S, Scount[d,x] - 1 = 0 means clean.
                # If x is NOT in S, Scount[d,x] = 0 means clean.
                # So if x is unsatisfied, Scount[d,x] - S_arr[x] > 0 for all d.
                # So there must be other S points.
                pass
            
            # Recompute Scount (expensive, but necessary for correctness)
            # Optimization: Only update Scount for lines affected by changes?
            # For now, full recompute.
            S_ba = S_arr.tobytes()
            Scount = _compute_Scount(S_ba)
            unsat = _check_and_fix(S_ba, Scount)
            repairs += 1
            
        if not unsat:
            # We have a valid S. Try to grow it.
            current_size = np.sum(S_arr)
            if current_size > best_size:
                best_size = current_size
                best_S = S_arr.tobytes()
                
            # Growth phase
            growth_steps = 0
            max_growth_steps = 1000
            while growth_steps < max_growth_steps:
                if time.time() > deadline - 10: break
                
                # Pick a random point not in S
                # To make this efficient, maintain a list of available points?
                # Or just pick random and check if in S.
                q = rng.randrange(_SLOTS)
                if S_arr[q] == 1:
                    continue
                
                # Check if adding q is safe
                # We need to check if any point x on lines through q becomes unsatisfied.
                # Specifically, for each d, let L = Line_d(q).
                # For each x in L, if x was satisfied ONLY by direction d (i.e. Scount[d,x] - S_arr[x] == 0 
                # and for all d' != d, Scount[d',x] - S_arr[x] > 0), then adding q to S 
                # increases Scount[d,x] by 1, making it > 0, so x loses witness d.
                # If x had no other witnesses, x becomes unsatisfied.
                
                safe = True
                # We need to know if x has other witnesses.
                # This requires checking all 50 directions for each neighbor.
                # Too slow per step.
                
                # Alternative: Just add q, recompute, and if invalid, revert.
                # This is "optimistic" local search.
                S_arr[q] = 1
                S_ba = S_arr.tobytes()
                Scount_new = _compute_Scount(S_ba)
                unsat_new = _check_and_fix(S_ba, Scount_new)
                
                if not unsat_new:
                    # Valid! Keep it.
                    Scount = Scount_new
                    unsat = unsat_new
                    growth_steps += 1
                else:
                    # Invalid, revert
                    S_arr[q] = 0
                    S_ba = S_arr.tobytes()
                    # Scount and unsat remain from previous valid state
                    growth_steps += 1 # count as a step to avoid infinite loop on same q?
                    # Actually, we should probably skip this q for a while or just move on.
                    
                if time.time() > deadline - 10: break

    return best_S, best_size

def nikodym_set() -> np.ndarray:
    deadline = time.time() + 580  # safety margin
    
    # Use Local Search
    S, size = _local_search(deadline)
    
    # Verify validity just in case
    if not _is_valid(bytes(S)):
        # Fallback to empty S (valid)
        S = bytearray(_SLOTS)
    
    return _to_array(bytes(S))

