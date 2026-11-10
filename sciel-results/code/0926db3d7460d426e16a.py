import time
import random

N = 256
DEADLINE = 55.0  # Safety margin for 90s limit

def construct():
    start_time = time.time()
    
    # Precompute all 3-APs
    three_aps = []
    # aps_by_elem[v] = list of bitmasks of the other 2 elements
    aps_by_elem = [[] for _ in range(N + 1)]
    
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            three_aps.append((a, b, c))
            # Store masks for fast delta calculation
            # For a: others are b, c
            mask_bc = (1 << (b-1)) | (1 << (c-1))
            aps_by_elem[a].append(mask_bc)
            # For b: others are a, c
            mask_ac = (1 << (a-1)) | (1 << (c-1))
            aps_by_elem[b].append(mask_ac)
            # For c: others are a, b
            mask_ab = (1 << (a-1)) | (1 << (b-1))
            aps_by_elem[c].append(mask_ab)
            
    # Precompute 4-APs: map each element to list of bitmasks of the other 3 elements
    four_aps_by_v = [[] for _ in range(N + 1)]
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # For each element, the "others" are the remaining 3
            # a: others b, c, e
            mask_bce = (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            four_aps_by_v[a].append(mask_bce)
            # b: others a, c, e
            mask_ace = (1 << (a-1)) | (1 << (c-1)) | (1 << (e-1))
            four_aps_by_v[b].append(mask_ace)
            # c: others a, b, e
            mask_abe = (1 << (a-1)) | (1 << (b-1)) | (1 << (e-1))
            four_aps_by_v[c].append(mask_abe)
            # e: others a, b, c
            mask_abc = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            four_aps_by_v[e].append(mask_abc)

    def time_left():
        return DEADLINE - (time.time() - start_time)

    # Greedy construction using bitmasks
    def greedy_construct():
        S = 0
        for x in range(1, N + 1):
            bit = 1 << (x - 1)
            # Check if adding x creates a 4-AP
            # A 4-AP containing x is formed if the other 3 are already in S
            valid = True
            for mask in four_aps_by_v[x]:
                if (S & mask) == mask:
                    valid = False
                    break
            if valid:
                S |= bit
        return S

    S = greedy_construct()
    
    # Calculate initial score
    def count_3aps(S):
        count = 0
        # Iterate through all 3-APs
        # We can just check each 3-AP tuple
        # Or use the aps_by_elem? No, that counts per element, might double count?
        # 3-AP (a,b,c) is counted once.
        # Using aps_by_elem: each 3-AP appears in 3 lists.
        # So sum of counts in aps_by_elem / 3?
        # No, let's just iterate the list of 3-APs, it's ~10k items.
        for a, b, c in three_aps:
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            if (S & mask) == mask:
                count += 1
        return count

    current_score = count_3aps(S)
    best_S = S
    best_score = current_score
    
    # Helper to get list of set bits
    def get_set_bits(S):
        bits = []
        temp = S
        i = 0
        while temp:
            if temp & 1:
                bits.append(i + 1)
            temp >>= 1
            i += 1
        return bits

    # Helper to get list of unset bits
    def get_unset_bits(S):
        bits = []
        full_mask = (1 << N) - 1
        unset = full_mask ^ S
        temp = unset
        i = 0
        while temp:
            if temp & 1:
                bits.append(i + 1)
            temp >>= 1
            i += 1
        return bits

    no_improve_rounds = 0
    
    while time_left() > 1.0:
        elements = get_set_bits(S)
        non_elements = get_unset_bits(S)
        
        if not elements or not non_elements:
            break
        
        random.shuffle(elements)
        random.shuffle(non_elements)
        
        improved = False
        for u in elements:
            if time_left() < 1.0:
                break
            u_bit = 1 << (u - 1)
            for v in non_elements:
                if time_left() < 1.0:
                    break
                v_bit = 1 << (v - 1)
                
                # Check validity of swap u -> v
                # Invalid if there is a 4-AP containing v where the other 3 are in S'
                # S' = S \ {u} U {v}
                # Other 3 in S' iff Other 3 in S AND u not in Other 3
                valid = True
                for mask in four_aps_by_v[v]:
                    # If mask is subset of S, and u is NOT in mask, then the AP is formed in S'
                    if (S & mask) == mask:
                        if not (mask & u_bit):
                            valid = False
                            break
                if not valid:
                    continue
                
                # Calculate Delta
                gained = 0
                # 3-APs containing v that become complete
                # Complete in S' iff others in S'
                # Others in S' iff Others in S AND u not in Others
                for mask in aps_by_elem[v]:
                    if (S & mask) == mask:
                        if not (mask & u_bit):
                            gained += 1
                
                lost = 0
                # 3-APs containing u that were complete in S
                # Complete in S iff others in S
                # They are lost because u is removed.
                # (v is not in 'others' because 'others' are the other 2 elements of the AP containing u.
                # If v was one of them, it wasn't in S, so the AP wasn't complete in S.)
                for mask in aps_by_elem[u]:
                    if (S & mask) == mask:
                        lost += 1
                        
                delta = gained - lost
                if delta > 0:
                    S = (S & ~u_bit) | v_bit
                    current_score += delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S
                    break
            if improved:
                break
        
        if not improved:
            no_improve_rounds += 1
            # Random kick
            if no_improve_rounds >= 3:
                for attempt in range(20):
                    if time_left() < 1.0:
                        break
                    u = random.choice(elements)
                    v = random.choice(non_elements)
                    u_bit = 1 << (u - 1)
                    v_bit = 1 << (v - 1)
                    
                    valid = True
                    for mask in four_aps_by_v[v]:
                        if (S & mask) == mask:
                            if not (mask & u_bit):
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    gained = 0
                    for mask in aps_by_elem[v]:
                        if (S & mask) == mask:
                            if not (mask & u_bit):
                                gained += 1
                    lost = 0
                    for mask in aps_by_elem[u]:
                        if (S & mask) == mask:
                            lost += 1
                            
                    delta = gained - lost
                    if delta >= -3:
                        S = (S & ~u_bit) | v_bit
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_S = S
                        no_improve_rounds = 0
                        break
                if no_improve_rounds >= 3:
                    no_improve_rounds = 0

    # Convert best_S bitmask to sorted list
    result = get_set_bits(best_S)
    return sorted(result)
