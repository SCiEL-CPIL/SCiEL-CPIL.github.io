"""Construct a 12-state binary deterministic synchronizing automaton with a long reset word."""
import time
import random
from collections import deque

N = 12

def _threshold(a, b):
    """BFS on subset automaton to find shortest reset word length. Returns -1 if not synchronizing."""
    start = frozenset(range(N))
    if len(start) == 1:
        return 0
    visited = {start: 0}
    q = deque([start])
    while q:
        curr = q.popleft()
        d = visited[curr]
        if len(curr) == 1:
            return d
        # apply a
        na = frozenset(a[i] for i in curr)
        if na not in visited:
            visited[na] = d + 1
            q.append(na)
        # apply b
        nb = frozenset(b[i] for i in curr)
        if nb not in visited:
            visited[nb] = d + 1
            q.append(nb)
    return -1

def _try_automaton(seed, time_limit=80.0):
    """Try to build a good automaton with given random seed. Returns (a, b, threshold) or None."""
    start_time = time.time()
    rng = random.Random(seed)
    
    # Strategy: Build automaton with a being a long cycle (permutation) and b being a 
    # carefully chosen map that gradually collapses the cycle.
    # For N=12, a = cycle 0->1->2->...->11->0
    a = [(i + 1) % N for i in range(N)]
    
    # b should map most states forward in a way that creates a long chain of subset sizes
    # before collapsing to a single state.
    
    # Approach: Use a "pinwheel" or "de Bruijn-like" structure.
    # Known best for N=12 is around 30-40 threshold based on literature.
    
    # Let's try a family: b maps i -> (i + k) % N for most i, but with some exceptions.
    # Actually, a better approach: start with a as a permutation (cycle), and b as a 
    # map that is "almost" a cycle but has one "defect" that causes gradual contraction.
    
    # Family 1: a = cycle, b = shift by 1 except at one position where it maps to a fixed point
    # This gives threshold ~ N*(N-1)/2 which for N=12 is 66. But let's verify.
    
    # Actually, the classic "pinwheel" automaton: a is a cycle, b maps i -> i+1 except 
    # b(0) = 0. Let's check this.
    # a: i -> i+1 mod N
    # b: i -> i+1 mod N for i != 0, b(0) = 0
    
    # From full set S = {0,...,N-1}:
    # S under a: {0,...,N-1} (same, it's a permutation)
    # S under b: {0, 1, 2, ..., N-1} (also same since b is almost a cycle)
    # Hmm, that doesn't contract.
    
    # Let me think differently. The known construction for long reset words uses:
    # a: a permutation (say a cycle)
    # b: a map that is identity on most states but maps one state to a different one,
    #     creating a "slow contraction" pattern.
    
    # Better family: 
    # a: i -> (i+1) % N  (the cycle)
    # b: i -> i for i != N-1, and b(N-1) = 0
    # This is the "de Bruijn" type. Let's analyze:
    # From {0,1,...,N-1} under b: {0,1,...,N-1} (b(N-1)=0, but 0 is already in range)
    # Actually b maps {0..N-1} to {0,1,...,N-2,0} = {0,...,N-2}. Size N-1.
    # Then {0,...,N-2} under a: {1,...,N-1,0} = {0,...,N-1}. Back to full!
    # So this oscillates. Not good.
    
    # Let me try: a = cycle, b = i -> (i-1) % N for most, with one exception.
    # Or better, use the known "Černý's worst case" construction.
    
    # For N states, the pinwheel automaton has reset word length (N-1)(N-2)/2.
    # For N=12: 11*10/2 = 55.
    
    # Pinwheel: a is the cycle 0->1->...->N-1->0.
    # b: 0->0, 1->0, 2->1, 3->2, ..., N-1->N-2
    # i.e., b(i) = max(0, i-1) for i>0, b(0)=0.
    
    # Let's verify: b = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] for N=12
    # a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
    
    # From {0,1,...,11}:
    # under b: {0,1,2,...,10} (size 11)
    # under a: {1,2,...,11,0} = {0,...,11} (size 12, back to start)
    # Hmm, that's bad.
    
    # Let me reconsider. The correct pinwheel:
    # a: i -> (i+1) mod N
    # b: 0 -> 0, i -> i-1 for i > 0
    # So b = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Start: {0,...,11}
    # apply b: {0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10} = {0,...,10}
    # apply a: {1,...,11, 0} = {0,...,11} -- back to full!
    
    # So that doesn't work directly. Let me look up the actual construction.
    
    # The correct Černý worst-case: 
    # a: 0->1, 1->2, ..., N-2->N-1, N-1->0 (cycle)
    # b: 0->0, 1->0, 2->1, 3->2, ..., N-1->N-2
    # Wait, that's what I had. Let me trace more carefully for small N.
    
    # For N=3: a=[1,2,0], b=[0,0,1]
    # {0,1,2} ->b-> {0,0,1}={0,1} ->a-> {1,2,0}={0,1,2} -- cycle!
    
    # That's not right. Let me try the other direction:
    # b: 0->0, 1->1, 2->1 for N=3. So b=[0,1,1]
    # {0,1,2} ->b-> {0,1} ->a-> {1,2} ->b-> {1} -- done in 3 steps.
    
    # For N=4: a=[1,2,3,0], b=[0,1,2,2]
    # {0,1,2,3} ->b-> {0,1,2} ->a-> {1,2,3} ->b-> {1,2} ->a-> {2,3} ->b-> {2} -- 5 steps.
    
    # For N=5: a=[1,2,3,4,0], b=[0,1,2,3,3]
    # {0..4} ->b-> {0..3} ->a-> {1..4} ->b-> {1..3} ->a-> {2..4} ->b-> {2,3} ->a-> {3,4} ->b-> {3} -- 7 steps
    
    # Pattern: N=3:3, N=4:5, N=5:7. So 2N-3? For N=12: 21.
    
    # Hmm, that's not 55. Let me try a different b.
    
    # b: 0->0, 1->0, 2->1, 3->2, ..., N-1->N-2. For N=4: b=[0,0,1,2]
    # {0,1,2,3} ->b-> {0,1,2} ->a-> {1,2,3} ->b-> {0,1,2} -- cycle!
    
    # OK so the monotonic b (b(i)=min(i, something)) works. Let me just go with that family
    # and also try random perturbations.
    
    # Family: a = cycle, b = [0, 1, 2, ..., N-2, N-2] (last two map to N-2)
    # For N=12: b = [0,1,2,3,4,5,6,7,8,9,10,10]
    
    base_b = list(range(N-1)) + [N-2]
    b = base_b[:]
    
    thr = _threshold(a, b)
    if thr > 0:
        return a, b, thr
    
    # Try random perturbations of b
    best_thr = thr if thr > 0 else 0
    best_a, best_b = a[:], b[:]
    
    for _ in range(200):
        if time.time() - start_time > time_limit:
            break
        rng.shuffle(b)
        thr = _threshold(a, b)
        if thr > best_thr:
            best_thr = thr
            best_a, best_b = a[:], b[:]
    
    return best_a, best_b, best_thr

def construct():
    """Construct a 12-state binary deterministic synchronizing automaton with long reset word."""
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    best_a, best_b, best_thr = None, None, 0
    
    # Try several families of automata
    
    # Family 1: a = cycle, b = monotonic with last two mapping to same
    a1 = [(i + 1) % N for i in range(N)]
    b1 = list(range(N-1)) + [N-2]
    thr1 = _threshold(a1, b1)
    if thr1 > best_thr:
        best_a, best_b, best_thr = a1[:], b1[:], thr1
    
    if time.time() > deadline:
        return best_a, best_b
    
    # Family 2: a = cycle, b = identity on most, one "defect"
    a2 = [(i + 1) % N for i in range(N)]
    for defect in range(N):
        if time.time() > deadline:
            break
        b2 = list(range(N))
        b2[defect] = (defect + N // 2) % N  # map one state far away
        thr2 = _threshold(a2, b2)
        if thr2 > best_thr:
            best_a, best_b, best_thr = a2[:], b2[:], thr2
    
    if time.time() > deadline:
        return best_a, best_b
    
    # Family 3: Two different permutations for a and b
    # a = cycle, b = shift by 2 with one exception
    a3 = [(i + 1) % N for i in range(N)]
    for step in range(1, N):
        if time.time() > deadline:
            break
        b3 = [(i + step) % N for i in range(N)]
        # Add a defect
        for defect in range(N):
            if time.time() > deadline:
                break
            b3 = [(i + step) % N for i in range(N)]
            b3[defect] = (defect + 1) % N  # one state goes forward by 1
            thr3 = _threshold(a3, b3)
            if thr3 > best_thr:
                best_a, best_b, best_thr = a3[:], b3[:], thr3
    
    if time.time() > deadline:
        return best_a, best_b
    
    # Family 4: Random search with restarts
    rng = random.Random(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        # a is a random permutation
        a4 = list(range(N))
        rng.shuffle(a4)
        # b is a random function
        b4 = [rng.randint(0, N-1) for _ in range(N)]
        thr4 = _threshold(a4, b4)
        if thr4 > best_thr:
            best_a, best_b, best_thr = a4[:], b4[:], thr4
    
    if time.time() > deadline:
        return best_a, best_b
    
    # Family 5: Structured search - a is cycle, b has a specific structure
    # b maps most states to i+1 but creates a "bottleneck"
    a5 = [(i + 1) % N for i in range(N)]
    for bottleneck in range(N):
        if time.time() > deadline:
            break
        b5 = []
        for i in range(N):
            if i == bottleneck:
                b5.append(0)  # map to sink
            elif i + 1 == bottleneck:
                b5.append(0)  # neighbor also maps to sink
            else:
                b5.append((i + 1) % N)
        thr5 = _threshold(a5, b5)
        if thr5 > best_thr:
            best_a, best_b, best_thr = a5[:], b5[:], thr5
    
    # Ensure we return a valid synchronizing automaton
    if best_a is None or best_b is None or best_thr <= 0:
        # Fallback: simple synchronizing automaton
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        best_thr = _threshold(best_a, best_b)
    
    return best_a, best_b
