import time
import random

# EVOLVE-BLOCK-START
def construct():
    N = 512
    deadline = time.time() + 85.0  # 85s limit, 5s safety margin
    
    # 1. Precompute 4-APs
    # An AP is (a, d) such that a, a+d, a+2d, a+3d <= N
    # Represent AP as a tuple of 4 elements or just (a, d)
    # We need to map elements to APs
    aps = []
    elem_to_aps = [[] for _ in range(N + 1)]
    
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            ap_id = len(aps)
            aps.append(ap)
            for x in ap:
                elem_to_aps[x].append(ap_id)
                
    num_aps = len(aps)
    
    # 2. Greedy Construction
    def greedy():
        S = set()
        # We need to check if adding x creates a 4-AP
        # A 4-AP is created if there exist y, z, w in S such that {x, y, z, w} is a 4-AP
        # This is equivalent to: for all APs containing x, the other 3 elements are not all in S.
        # To make this fast, we can just check all APs containing x.
        for x in range(1, N + 1):
            can_add = True
            for ap_id in elem_to_aps[x]:
                ap = aps[ap_id]
                # Check if the other 3 elements are in S
                # Since x is being added, we check if the other 3 are in S
                other_3 = [y for y in ap if y != x]
                if all(y in S for y in other_3):
                    can_add = False
                    break
            if can_add:
                S.add(x)
        return S

    current_S = greedy()
    best_S = set(current_S)
    
    # 3. Local Search
    # State:
    # S: set of elements
    # count[ap_id]: number of elements of ap in S
    # blocked[u]: list of ap_ids that block u (i.e., count[ap_id]==3 and u is the missing element)
    # Actually, we just need to know if u is blocked.
    # u is blocked if there exists ap_id in elem_to_aps[u] such that count[ap_id] == 3 and u not in S.
    # Wait, if u in S, it's not "missing".
    # If u not in S, u is blocked if count[ap_id] == 3 for some ap_id containing u.
    # Note: if count[ap_id] == 4, the set is invalid. We must avoid this.
    # Our moves will ensure validity.
    
    def init_state(S):
        count = [0] * num_aps
        for x in S:
            for ap_id in elem_to_aps[x]:
                count[ap_id] += 1
        return count

    def is_addable(u, S, count):
        if u in S:
            return False
        for ap_id in elem_to_aps[u]:
            if count[ap_id] == 3:
                return False
        return True

    def try_add(u, S, count):
        # Add u to S and update count
        S.add(u)
        for ap_id in elem_to_aps[u]:
            count[ap_id] += 1

    def try_remove(v, S, count):
        # Remove v from S and update count
        S.remove(v)
        for ap_id in elem_to_aps[v]:
            count[ap_id] -= 1

    count = init_state(current_S)
    
    # Optimization: Precompute missing element for each AP given current S?
    # No, we can compute it on the fly.
    # For an AP with count 3, the missing element is the one in AP not in S.
    
    while time.time() < deadline:
        # Phase 1: Add all addable elements
        changed = True
        while changed:
            changed = False
            # Iterate over a copy of candidates to avoid modification during iteration
            # Candidates are elements not in S
            # We can just iterate 1..N
            for u in range(1, N + 1):
                if u not in S and is_addable(u, S, count):
                    try_add(u, S, count)
                    changed = True
                    if len(S) > len(best_S):
                        best_S = set(S)
        
        # Phase 2: If stuck, remove an element to unblock others
        # Check if we are stuck: no addable elements
        # If we just finished Phase 1 and changed was False initially, we are stuck.
        # But we need to check if there are ANY addable elements.
        # The loop above exits if no changes were made in a full pass.
        
        # Calculate unblocking potential for each v in S
        # v unblocks u if all APs blocking u contain v.
        # u is blocked by APs e where count[e]==3 and u in e and u not in S.
        
        # Let's find all blocked u's first
        blocked_us = []
        for u in range(1, N + 1):
            if u not in S:
                blockers = []
                for ap_id in elem_to_aps[u]:
                    if count[ap_id] == 3:
                        blockers.append(ap_id)
                if blockers:
                    blocked_us.append((u, blockers))
        
        if not blocked_us:
            # No blocked elements, and no addable elements?
            # If no addable and no blocked, then all u not in S are addable?
            # No, if u not in S and not blocked, it is addable.
            # So if we are stuck, there must be blocked elements.
            # Unless S is all of [1, N], which is impossible.
            break

        # For each v in S, count how many blocked u's it unblocks.
        # u is unblocked by v if all blockers of u contain v.
        # i.e., set(blockers) is subset of set(elem_to_aps[v])? No.
        # blockers are ap_ids. v is an element.
        # v is in ap_id if ap_id is in elem_to_aps[v].
        # So we check if all ap_ids in blockers are in elem_to_aps[v].
        
        # To speed this up, we can just pick a random v if the calculation is too slow.
        # Given N=512, |S|~200, |blocked_us|~200, |blockers|~small.
        # It should be fast enough.
        
        best_v = None
        best_unblocked_count = -1
        
        # Optimize: Only consider v's that are in at least one blocking AP
        # Actually, if v is not in any blocking AP, it unblocks 0.
        
        for v in list(S):
            v_aps = set(elem_to_aps[v])
            unblocked_count = 0
            for u, blockers in blocked_us:
                # Check if all blockers are in v_aps
                # blockers is a list of ap_ids
                if all(b in v_aps for b in blockers):
                    unblocked_count += 1
            if unblocked_count > best_unblocked_count:
                best_unblocked_count = unblocked_count
                best_v = v
        
        if best_v is None or best_unblocked_count == 0:
            # No single removal unblocks anything?
            # This can happen if blockers are spread out.
            # Pick a random v
            best_v = random.choice(list(S))
        
        try_remove(best_v, S, count)
        
        # Update best_S if current S is better? No, we removed one, so it's worse.
        # But it might lead to a better state later.
        
    return sorted(best_S)
# EVOLVE-BLOCK-END
