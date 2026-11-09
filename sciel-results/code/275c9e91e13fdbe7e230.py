"""Improved program: greedy + local search to maximize valid set size in 1..128."""
import time
import random

N = 128
DEADLINE = 85.0  # safety margin before 90s kill

def _is_valid(S_set):
    """Check no distinct a,b,c in S_set with a+b, a+c, b+c all in S_set."""
    lst = list(S_set)
    n = len(lst)
    # For each pair (a,b) with a<b, if a+b in S, then we must not have any c in S
    # (c distinct from a,b) such that a+c and b+c are in S.
    # Equivalently: for distinct a,b,c, not (a+b in S and a+c in S and b+c in S).
    # Efficient: for each pair (a,b) in S with a<b and a+b in S,
    #   the set of c that would violate is c in S, c!=a,b, with (a+c in S) and (b+c in S).
    # We can precompute for efficiency but n<=128 so O(n^3) is 2M ops, fine for a few calls.
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            s = a + b
            if s not in S_set:
                continue
            # need a third distinct c in S with a+c in S and b+c in S
            for k in range(n):
                c = lst[k]
                if c == a or c == b:
                    continue
                if (a + c) in S_set and (b + c) in S_set:
                    return False
    return True

def _can_add(S_set, x):
    """Can we add x to S_set while staying valid?"""
    # Adding x creates violations only if there exist distinct a,b in S_set
    # (a,b != x) such that with c=x, we have a+x, b+x, a+b all in S_set ∪ {x}.
    # Since x not in S_set yet, the sums involving x (a+x, b+x) must be in S_set
    # (not x itself, since x not in S_set and a+x > x typically but could equal? a+x=x => a=0 impossible).
    # And a+b must be in S_set ∪ {x} => a+b in S_set or a+b==x.
    # Also the triple (a,b,x) with a,b distinct from x and from each other.
    for a in S_set:
        for b in S_set:
            if a == b:
                continue
            # triple (a,b,x) distinct
            # need a+b, a+x, b+x all in (S_set ∪ {x})
            ab = a + b
            ax = a + x
            bx = b + x
            # a+x in S_set ∪ {x}: ax == x => a==0 no. so ax in S_set.
            if ax not in S_set:
                continue
            if bx not in S_set:
                continue
            if ab in S_set or ab == x:
                return False
    return True

def _greedy_construct():
    """Greedy: add elements in order of increasing 'cost' (forced exclusions)."""
    S_set = set()
    # candidates 1..128
    # cost of adding x given current S: number of elements that would become invalid? 
    # Simpler: just try to add in a heuristic order.
    # Order: larger numbers first (their sums go out of range, less likely to conflict)
    # but also we want to include as many as possible.
    # Try adding in descending order, skip if can't add.
    for x in range(N, 0, -1):
        if _can_add(S_set, x):
            S_set.add(x)
    return S_set

def _local_search(S_set, time_limit):
    """Local search: try adding, removing, swapping to increase size."""
    start = time.time()
    best_size = len(S_set)
    best_set = set(S_set)
    cur = set(S_set)
    
    # Precompute for speed: valid check via _can_add and a full check
    random.seed(42)
    
    def full_valid(s):
        return _is_valid(s)
    
    # First ensure current is valid
    if not full_valid(cur):
        # repair: remove elements until valid (shouldn't happen)
        lst = list(cur)
        random.shuffle(lst)
        for x in lst:
            if len(cur) <= 1:
                break
            cur.remove(x)
            if full_valid(cur):
                break
        best_set = set(cur)
        best_size = len(cur)
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() - start < time_limit and no_improve < max_no_improve:
        improved = False
        # Try adding any element
        for x in range(1, N+1):
            if x in cur:
                continue
            if _can_add(cur, x):
                cur.add(x)
                improved = True
                break
        if improved:
            if len(cur) > best_size:
                best_size = len(cur)
                best_set = set(cur)
            no_improve = 0
            continue
        
        # Try removing one element and adding two? Or single remove then add.
        # Try: remove x, then try to add as many as possible
        # Or swap: remove x, add y
        cur_lst = list(cur)
        random.shuffle(cur_lst)
        swap_found = False
        for x in cur_lst:
            if time.time() - start > time_limit:
                break
            trial = set(cur)
            trial.remove(x)
            # try to add some element y not in trial
            added = False
            for y in range(1, N+1):
                if y in trial:
                    continue
                if _can_add(trial, y):
                    trial.add(y)
                    added = True
                    break
            if added:
                # now try to add more
                for y in range(1, N+1):
                    if y in trial:
                        continue
                    if _can_add(trial, y):
                        trial.add(y)
                if len(trial) > len(cur):
                    cur = trial
                    swap_found = True
                    if len(cur) > best_size:
                        best_size = len(cur)
                        best_set = set(cur)
                    break
        if swap_found:
            no_improve = 0
            continue
        
        # Try pure removal then re-add (sometimes removing a 'bad' element allows more)
        # Actually, removal only decreases size, so only useful if followed by more additions.
        # We already tried remove+add. 
        no_improve += 1
    
    return best_set

def construct():
    start = time.time()
    # Greedy construction
    S_set = _greedy_construct()
    # Local search with time limit
    time_left = DEADLINE - time.time() - 2.0  # 2s safety margin
    if time_left > 1.0:
        S_set = _local_search(S_set, time_left)
    
    # Final validity check
    if not _is_valid(S_set):
        # Fallback to seed
        return list(range(N//2+1, N+1))
    
    # Ensure Python ints, list
    result = [int(x) for x in S_set]
    # Check distinct and in range
    assert len(set(result)) == len(result)
    assert all(1 <= x <= N for x in result)
    return result

# Quick self-test at import? No, keep it fast.
if __name__ == "__main__":
    t0 = time.time()
    r = construct()
    t1 = time.time()
    print(f"Size: {len(r)}, Time: {t1-t0:.2f}s")
    print(f"Valid: {_is_valid(set(r))}")
    print(r)
