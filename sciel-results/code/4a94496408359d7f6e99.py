"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9

# EVOLVE-BLOCK-START
def _greedy_fast(start, bit_order, deadline):
    """Fast greedy snake construction using O(1) exclusion-set maintenance.
    
    Instead of checking distance from all path vertices each step, we maintain
    a set of 'excluded' vertices (those at distance 1 from any non-last path
    vertex). A candidate v is valid iff v is not in excluded and not the
    previous vertex.
    
    Returns the path as a list of ints.
    """
    path = [start]
    prev = start
    # excluded[v] means v is at Hamming distance 1 from some path vertex
    # that is NOT the most recently added one.
    # We maintain this incrementally.
    excluded = set()
    # When we add vertex v to path, all neighbors of v become excluded
    # EXCEPT the one we came from (prev), which is already in the path.
    # But actually, we need to be careful: the "non-last" condition means
    # that when we append v, the new "last" is v, and all previous vertices
    # are "non-last". So v's neighbors (except prev) must be excluded.
    # Also, prev was the last before; now prev becomes non-last, so prev's
    # neighbors (except the one before prev) need to be excluded too.
    # 
    # Simpler approach: maintain a count of how many path vertices (excluding
    # the current last) are at distance 1 from each candidate. But that's O(N)
    # per step.
    #
    # Alternative: For each candidate v at the current step, check if v is
    # at distance 1 from any path vertex except the last. We can maintain
    # a set `blocked` of vertices that are at distance 1 from at least one
    # non-last path vertex. When we add a new vertex v:
    #   - The old last (prev) becomes non-last, so all neighbors of prev 
    #     (except the vertex before prev, which is already non-last or was
    #     the last) get added to blocked.
    #   - The new last v: its neighbors do NOT get added to blocked yet
    #     (they will be added when v stops being the last).
    #   - But we also need: v itself must not be in blocked, and v must not
    #     be prev (already in path).
    #
    # Let me think again. At each step, the current last is `cur`. We want
    # to find a neighbor v of cur such that:
    #   1. v is not in path
    #   2. v is not at distance 1 from any path vertex except cur
    #
    # Let `blocked` = set of vertices that are at distance 1 from at least
    # one path vertex other than the current last.
    # Then v is valid iff v not in blocked and v not in path.
    #
    # When we move from cur to v (v becomes new last):
    #   - cur was last, now cur is non-last. So all neighbors of cur that
    #     are not already blocked get added to blocked. But wait, v is a
    #     neighbor of cur and v is now the last, so v should NOT be in
    #     blocked. Actually v is in path now, so it's excluded by "not in
    #     path" check. But v's other neighbors: they are at distance 1 from
    #     cur (now non-last), so they should be in blocked.
    #   - v is now last. Its neighbors should NOT be in blocked (yet).
    #
    # So the update is:
    #   - Add all neighbors of cur to blocked (except v, which is now last
    #     and in path).
    #   - v is now in path.
    #
    # But we also need to handle the initial state. Initially path = [start],
    # last = start, and there are no non-last vertices, so blocked = {}.
    #
    # Let's also track `in_path` as a set for O(1) membership.

    in_path = {start}
    blocked = set()
    
    while True:
        if time.time() > deadline:
            break
        cur = path[-1]
        found = False
        for b in bit_order:
            v = cur ^ (1 << b)
            if v in in_path:
                continue
            if v in blocked:
                continue
            # v is valid
            # Update: cur becomes non-last, so add cur's neighbors to blocked
            for b2 in range(N):
                nb = cur ^ (1 << b2)
                if nb != v:  # v is now last, don't block it
                    blocked.add(nb)
            # v becomes last
            in_path.add(v)
            path.append(v)
            found = True
            break
        if not found:
            break
    
    return path

def _greedy_with_lookahead(start, bit_order, deadline):
    """Greedy with 1-step lookahead: at each step, pick the neighbor that
    leaves the most options for the next step."""
    path = [start]
    in_path = {start}
    blocked = set()
    
    while True:
        if time.time() > deadline:
            break
        cur = path[-1]
        candidates = []
        for b in bit_order:
            v = cur ^ (1 << b)
            if v in in_path or v in blocked:
                continue
            candidates.append(v)
        
        if not candidates:
            break
        
        if len(candidates) == 1:
            v = candidates[0]
        else:
            # Pick candidate that maximizes number of valid next moves
            best_v = None
            best_count = -1
            for v in candidates:
                # Simulate: if we go to v, what's the count of valid next?
                # cur becomes non-last, so cur's neighbors (except v) get blocked.
                # v becomes last. Valid next from v: neighbors of v not in
                # (in_path ∪ {v} ∪ (blocked ∪ neighbors of cur except v))
                new_blocked = set(blocked)
                for b2 in range(N):
                    nb = cur ^ (1 << b2)
                    if nb != v:
                        new_blocked.add(nb)
                count = 0
                for b2 in range(N):
                    nb = v ^ (1 << b2)
                    if nb in in_path:
                        continue
                    if nb == cur:
                        continue
                    if nb in new_blocked:
                        continue
                    count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            v = best_v
        
        # Commit to v
        for b2 in range(N):
            nb = cur ^ (1 << b2)
            if nb != v:
                blocked.add(nb)
        in_path.add(v)
        path.append(v)
    
    return path

def construct():
    """Construct the longest snake in Q_9 using fast greedy with many restarts."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_snake = []
    best_len = 0
    
    # Phase 1: Deterministic greedy from all 512 starting vertices
    for start in range(512):
        if time.time() > deadline:
            break
        bit_order = list(range(N))
        snake = _greedy_fast(start, bit_order, deadline)
        if len(snake) > best_len:
            best_len = len(snake)
            best_snake = snake
    
    # Phase 2: Randomized greedy with random starting vertices and bit orders
    trial = 0
    while time.time() < deadline - 10.0:
        start = random.randint(0, 511)
        bit_order = list(range(N))
        random.shuffle(bit_order)
        snake = _greedy_fast(start, bit_order, deadline)
        if len(snake) > best_len:
            best_len = len(snake)
            best_snake = snake
        trial += 1
        # Check time more frequently
        if trial % 100 == 0 and time.time() > deadline - 10.0:
            break
    
    # Phase 3: Lookahead greedy on best result's neighborhood
    if time.time() < deadline - 5.0:
        for start in range(512):
            if time.time() > deadline - 5.0:
                break
            bit_order = list(range(N))
            snake = _greedy_with_lookahead(start, bit_order, deadline)
            if len(snake) > best_len:
                best_len = len(snake)
                best_snake = snake
    
    # Phase 4: More randomized lookahead
    while time.time() < deadline - 2.0:
        start = random.randint(0, 511)
        bit_order = list(range(N))
        random.shuffle(bit_order)
        snake = _greedy_with_lookahead(start, bit_order, deadline)
        if len(snake) > best_len:
            best_len = len(snake)
            best_snake = snake
    
    return best_snake
# EVOLVE-BLOCK-END
