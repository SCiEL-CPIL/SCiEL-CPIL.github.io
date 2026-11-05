import time
import random
import math
from collections import defaultdict

N = 256
MAX_EACH = 64
DEADLINE_MARGIN = 10.0  # safety margin before 90s limit


def _is_sidon(S):
    """Check if S is a Sidon set (pair sums including a+a unique)."""
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _diffs(S):
    """Nonzero difference set of S."""
    d = set()
    for i in range(len(S)):
        for j in range(len(S)):
            if i != j:
                d.add(S[i] - S[j])
    return d


def _valid_pair(A, B):
    """Check both Sidon and disjoint nonzero differences."""
    if not A or not B:
        return False
    if not _is_sidon(A) or not _is_sidon(B):
        return False
    dA = _diffs(A)
    dB = _diffs(B)
    return len(dA & dB) == 0


def _init_state(A, B):
    """Initialize sum and diff counters for A and B."""
    A_sums = defaultdict(int)
    B_sums = defaultdict(int)
    A_diffs = defaultdict(int)
    B_diffs = defaultdict(int)
    
    for i in range(len(A)):
        for j in range(i, len(A)):
            A_sums[A[i] + A[j]] += 1
        for j in range(len(A)):
            if i != j:
                A_diffs[A[i] - A[j]] += 1
                
    for i in range(len(B)):
        for j in range(i, len(B)):
            B_sums[B[i] + B[j]] += 1
        for j in range(len(B)):
            if i != j:
                B_diffs[B[i] - B[j]] += 1
                
    return A_sums, B_sums, A_diffs, B_diffs

def _greedy_build(candidates_order, prefer_A_first=True, deadline=None):
    """Greedy build: for each candidate in order, try adding to A or B."""
    A = []
    B = []
    A_sums = set()
    B_sums = set()
    A_diffs = set()
    B_diffs = set()

    def can_add_to_A(x):
        for a in A:
            s = x + a
            if s in A_sums:
                return False
        if 2 * x in A_sums:
            return False
        for a in A:
            d = x - a
            if d in B_diffs:
                return False
        return True

    def can_add_to_B(x):
        for b in B:
            s = x + b
            if s in B_sums:
                return False
        if 2 * x in B_sums:
            return False
        for b in B:
            d = x - b
            if d in A_diffs:
                return False
        return True

    for x in candidates_order:
        if deadline is not None and time.time() > deadline:
            break
        okA = can_add_to_A(x)
        okB = can_add_to_B(x)
        if okA and okB:
            # Prefer to balance sizes; add to smaller
            if len(A) <= len(B) and prefer_A_first:
                A.append(x)
                for a in A[:-1]:
                    A_sums.add(x + a)
                    A_diffs.add(x - a)
                    A_diffs.add(a - x)
                A_sums.add(2 * x)
            else:
                B.append(x)
                for b in B[:-1]:
                    B_sums.add(x + b)
                    B_diffs.add(x - b)
                    B_diffs.add(b - x)
                B_sums.add(2 * x)
        elif okA:
            A.append(x)
            for a in A[:-1]:
                A_sums.add(x + a)
                A_diffs.add(x - a)
                A_diffs.add(a - x)
            A_sums.add(2 * x)
        elif okB:
            B.append(x)
            for b in B[:-1]:
                B_sums.add(x + b)
                B_diffs.add(x - b)
                B_diffs.add(b - x)
            B_sums.add(2 * x)
    return A, B

def _local_search(A, B, max_iters=2000, deadline=None):
    """Try to improve by swapping elements between sets or adding new ones."""
    A = list(A)
    B = list(B)
    
    A_sums, B_sums, A_diffs, B_diffs = _init_state(A, B)
    
    def score():
        return len(A) + len(B)
        
    best_A, best_B = list(A), list(B)
    best_score = score()
    cur_score = best_score

    all_vals = set(range(1, N + 1))
    used = set(A) | set(B)
    unused_list = list(all_vals - used)

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        move_type = random.random()
        
        # Helper to check if x can be added to S (given other set's diffs)
        # This is done inline for speed
        
        if move_type < 0.3:
            # Add new element to A or B
            if not unused_list:
                continue
            x = random.choice(unused_list)
            target = A if random.random() < 0.5 else B
            target_sums = A_sums if target is A else B_sums
            target_diffs = A_diffs if target is A else B_diffs
            other_diffs = B_diffs if target is A else A_diffs
            
            ok = True
            # Check Sidon
            if target_sums[2*x] > 0:
                ok = False
            else:
                for y in target:
                    if target_sums[x+y] > 0:
                        ok = False
                        break
            if ok:
                # Check Diffs
                for y in target:
                    if other_diffs[x-y] > 0 or other_diffs[y-x] > 0:
                        ok = False
                        break
                        
            if ok:
                # Apply
                target.append(x)
                target_sums[2*x] += 1
                for y in target[:-1]:
                    target_sums[x+y] += 1
                    target_diffs[x-y] += 1
                    target_diffs[y-x] += 1
                unused_list.remove(x)
                used.add(x)
                
                s = score()
                if s > best_score:
                    best_score = s
                    best_A, best_B = list(A), list(B)
                    cur_score = s
            # If not ok, do nothing (x remains in unused_list)

        elif move_type < 0.5:
            # Move element from A to B
            if len(A) < 2:
                continue
            a = random.choice(A)
            
            # Check if a can be added to B (assuming it's removed from A first? 
            # Actually, moving A->B means: remove from A, add to B.
            # The diff check for adding to B depends on A's diffs *after* removal.
            # This is complex. 
            # Alternative: Just try the move and validate? 
            # Since we have incremental state, we can:
            # 1. Remove a from A (update state)
            # 2. Check if a can be added to B (using updated A_diffs and B_sums/diffs)
            # 3. If yes, add to B.
            # 4. If no, revert removal from A.
            
            # Remove a from A
            A.remove(a)
            A_sums[2*a] -= 1
            for y in A:
                A_sums[a+y] -= 1
                A_diffs[a-y] -= 1
                A_diffs[y-a] -= 1
            # Cleanup zeros to keep sets small? Or just check >0.
            
            # Check if a can be added to B
            ok = True
            if B_sums[2*a] > 0:
                ok = False
            else:
                for y in B:
                    if B_sums[a+y] > 0:
                        ok = False
                        break
            if ok:
                for y in B:
                    if A_diffs[a-y] > 0 or A_diffs[y-a] > 0:
                        ok = False
                        break
                        
            if ok:
                # Add a to B
                if a not in B:
                    B.append(a)
                    B_sums[2*a] += 1
                    for y in B[:-1]:
                        B_sums[a+y] += 1
                        B_diffs[a-y] += 1
                        B_diffs[y-a] += 1
                else:
                    # If a was already in B, we just removed it from A.
                    # Wait, "Move element from A to B" implies it wasn't in B?
                    # The problem says A and B can overlap.
                    # If a is in A and B, removing from A leaves it in B.
                    # This effectively reduces |A| and keeps |B| same?
                    # No, the move is "Move a from A to B".
                    # If a is in both, removing from A is just a removal from A.
                    # The "Add to B" part is redundant if it's already there.
                    # Let's simplify: Only consider moves if a not in B.
                    # If a in B, this is just a "Remove from A" move.
                    # Let's handle "Remove from A" separately or just allow it.
                    # If a in B, we just removed it from A.
                    # Score change: -1.
                    # We only accept if score > best_score. So -1 moves are never accepted.
                    # So we can skip this case.
                    pass
                    
                s = score()
                if s > best_score:
                    best_score = s
                    best_A, best_B = list(A), list(B)
                    cur_score = s
            else:
                # Revert removal from A
                A.append(a)
                A_sums[2*a] += 1
                for y in A[:-1]:
                    A_sums[a+y] += 1
                    A_diffs[a-y] += 1
                    A_diffs[y-a] += 1
                    
        elif move_type < 0.7:
            # Move element from B to A
            # Symmetric to above
            if len(B) < 2:
                continue
            b = random.choice(B)
            
            B.remove(b)
            B_sums[2*b] -= 1
            for y in B:
                B_sums[b+y] -= 1
                B_diffs[b-y] -= 1
                B_diffs[y-b] -= 1
                
            ok = True
            if A_sums[2*b] > 0:
                ok = False
            else:
                for y in A:
                    if A_sums[b+y] > 0:
                        ok = False
                        break
            if ok:
                for y in A:
                    if B_diffs[b-y] > 0 or B_diffs[y-b] > 0:
                        ok = False
                        break
                        
            if ok:
                if b not in A:
                    A.append(b)
                    A_sums[2*b] += 1
                    for y in A[:-1]:
                        A_sums[b+y] += 1
                        A_diffs[b-y] += 1
                        A_diffs[y-b] += 1
                s = score()
                if s > best_score:
                    best_score = s
                    best_A, best_B = list(A), list(B)
                    cur_score = s
            else:
                B.append(b)
                B_sums[2*b] += 1
                for y in B[:-1]:
                    B_sums[b+y] += 1
                    B_diffs[b-y] += 1
                    B_diffs[y-b] += 1
                    
        else:
            # Swap a from A, b from B
            if len(A) < 2 or len(B) < 2:
                continue
            a = random.choice(A)
            b = random.choice(B)
            if a == b:
                continue
                
            # Remove a from A
            A.remove(a)
            A_sums[2*a] -= 1
            for y in A:
                A_sums[a+y] -= 1
                A_diffs[a-y] -= 1
                A_diffs[y-a] -= 1
                
            # Remove b from B
            B.remove(b)
            B_sums[2*b] -= 1
            for y in B:
                B_sums[b+y] -= 1
                B_diffs[b-y] -= 1
                B_diffs[y-b] -= 1
                
            # Check if b can be added to A
            okA = True
            if A_sums[2*b] > 0:
                okA = False
            else:
                for y in A:
                    if A_sums[b+y] > 0:
                        okA = False
                        break
            if okA:
                for y in A:
                    if B_diffs[b-y] > 0 or B_diffs[y-b] > 0:
                        okA = False
                        break
                        
            # Check if a can be added to B
            okB = True
            if B_sums[2*a] > 0:
                okB = False
            else:
                for y in B:
                    if B_sums[a+y] > 0:
                        okB = False
                        break
            if okB:
                for y in B:
                    if A_diffs[a-y] > 0 or A_diffs[y-a] > 0:
                        okB = False
                        break
                        
            if okA and okB:
                # Add b to A
                A.append(b)
                A_sums[2*b] += 1
                for y in A[:-1]:
                    A_sums[b+y] += 1
                    A_diffs[b-y] += 1
                    A_diffs[y-b] += 1
                    
                # Add a to B
                B.append(a)
                B_sums[2*a] += 1
                for y in B[:-1]:
                    B_sums[a+y] += 1
                    B_diffs[a-y] += 1
                    B_diffs[y-a] += 1
                    
                s = score()
                if s > best_score:
                    best_score = s
                    best_A, best_B = list(A), list(B)
                    cur_score = s
            else:
                # Revert
                A.append(a)
                A_sums[2*a] += 1
                for y in A[:-1]:
                    A_sums[a+y] += 1
                    A_diffs[a-y] += 1
                    A_diffs[y-a] += 1
                    
                B.append(b)
                B_sums[2*b] += 1
                for y in B[:-1]:
                    B_sums[b+y] += 1
                    B_diffs[b-y] += 1
                    B_diffs[y-b] += 1

    return best_A, best_B


def construct():
    deadline = time.time() + (90.0 - DEADLINE_MARGIN)
    best_size = 0
    best_sets = ([], [])

    # Strategy 1: Deterministic greedy with different orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        [x for x in range(1, N + 1) if x % 2 == 1] + [x for x in range(1, N + 1) if x % 2 == 0],
        [x for x in range(N, 0, -1) if x % 2 == 0] + [x for x in range(N, 0, -1) if x % 2 == 1],
        list(range(1, 65)) + list(range(65, 129)) + list(range(129, 193)) + list(range(193, 257)),
        list(range(193, 257)) + list(range(129, 193)) + list(range(65, 129)) + list(range(1, 65)),
        [x for x in range(1, N + 1) if x % 3 != 0] + [x for x in range(1, N + 1) if x % 3 == 0],
        [x for x in range(N, 0, -1) if x % 3 != 0] + [x for x in range(N, 0, -1) if x % 3 == 0],
    ]

    for order in orderings:
        if time.time() > deadline:
            break
        A, B = _greedy_build(order, deadline=deadline)
        if _valid_pair(A, B):
            s = len(A) + len(B)
            if s > best_size:
                best_size = s
                best_sets = (list(A), list(B))
        A2, B2 = _greedy_build(order, prefer_A_first=False, deadline=deadline)
        if _valid_pair(A2, B2):
            s = len(A2) + len(B2)
            if s > best_size:
                best_size = s
                best_sets = (list(A2), list(B2))

    # Strategy 2: Randomized greedy with multiple restarts
    rng = random.Random(42)
    num_random_tries = 2000 # Increased due to faster local search
    for t in range(num_random_tries):
        if time.time() > deadline:
            break
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        A, B = _greedy_build(candidates, deadline=deadline)
        if _valid_pair(A, B):
            s = len(A) + len(B)
            if s > best_size:
                best_size = s
                best_sets = (list(A), list(B))
            # Local search on promising solutions
            if s >= best_size - 3:
                A_ls, B_ls = _local_search(A, B, max_iters=5000, deadline=deadline)
                if _valid_pair(A_ls, B_ls):
                    s2 = len(A_ls) + len(B_ls)
                    if s2 > best_size:
                        best_size = s2
                        best_sets = (list(A_ls), list(B_ls))

    # Strategy 3: Local search on the best found
    if best_sets[0] and best_sets[1]:
        A_ls, B_ls = _local_search(best_sets[0], best_sets[1], max_iters=20000, deadline=deadline)
        if _valid_pair(A_ls, B_ls):
            s = len(A_ls) + len(B_ls)
            if s > best_size:
                best_size = s
                best_sets = (list(A_ls), list(B_ls))

    # Fallback: ensure non-empty valid pair
    if not best_sets[0] or not best_sets[1]:
        best_sets = ([1, 10], [2, 20])
        best_size = 4

    # Final validation
    if not _valid_pair(best_sets[0], best_sets[1]):
        best_sets = ([1, 10], [2, 20])

    return [list(best_sets[0]), list(best_sets[1])]
