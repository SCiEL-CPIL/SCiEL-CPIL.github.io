import time
import random

N = 256
DEADLINE = 58.0  # Safety margin for 90s limit

def construct():
    start_time = time.time()
    
    # Precompute all 3-APs and 4-APs with bitmasks
    three_aps = []
    aps_by_elem = [[] for _ in range(N + 1)]
    
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            three_aps.append((a, b, c))
            mask_bc = (1 << (b-1)) | (1 << (c-1))
            aps_by_elem[a].append(mask_bc)
            mask_ac = (1 << (a-1)) | (1 << (c-1))
            aps_by_elem[b].append(mask_ac)
            mask_ab = (1 << (a-1)) | (1 << (b-1))
            aps_by_elem[c].append(mask_ab)
            
    four_aps_by_v = [[] for _ in range(N + 1)]
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask_bce = (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            four_aps_by_v[a].append(mask_bce)
            mask_ace = (1 << (a-1)) | (1 << (c-1)) | (1 << (e-1))
            four_aps_by_v[b].append(mask_ace)
            mask_abe = (1 << (a-1)) | (1 << (b-1)) | (1 << (e-1))
            four_aps_by_v[c].append(mask_abe)
            mask_abc = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            four_aps_by_v[e].append(mask_abc)

    def time_left():
        return DEADLINE - (time.time() - start_time)

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

    def count_3aps(S):
        count = 0
        for a, b, c in three_aps:
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            if (S & mask) == mask:
                count += 1
        return count

    # Greedy construction
    def greedy_construct():
        S = 0
        for x in range(1, N + 1):
            bit = 1 << (x - 1)
            valid = True
            for mask in four_aps_by_v[x]:
                if (S & mask) == mask:
                    valid = False
                    break
            if valid:
                S |= bit
        return S

    S = greedy_construct()
    current_score = count_3aps(S)
    best_S = S
    best_score = current_score
    
    # Local search with swaps
    no_improve_rounds = 0
    
    while time_left() > 2.0:
        elements = get_set_bits(S)
        non_elements = get_unset_bits(S)
        
        if not elements or not non_elements:
            break
        
        random.shuffle(elements)
        random.shuffle(non_elements)
        
        improved = False
        for u in elements:
            if time_left() < 2.0:
                break
            u_bit = 1 << (u - 1)
            for v in non_elements:
                if time_left() < 2.0:
                    break
                v_bit = 1 << (v - 1)
                
                # Check validity of swap u -> v
                valid = True
                for mask in four_aps_by_v[v]:
                    if (S & mask) == mask:
                        if not (mask & u_bit):
                            valid = False
                            break
                if not valid:
                    continue
                
                # Calculate Delta
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
            # Random kick to escape local optima
            if no_improve_rounds >= 3:
                kick_success = False
                for attempt in range(30):
                    if time_left() < 2.0:
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
                    if delta >= -5:
                        S = (S & ~u_bit) | v_bit
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_S = S
                        kick_success = True
                        break
                if kick_success:
                    no_improve_rounds = 0
                else:
                    no_improve_rounds = 0

    # Convert best_S bitmask to sorted list
    result = get_set_bits(best_S)
    return sorted(result)
