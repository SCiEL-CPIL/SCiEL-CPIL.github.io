import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute 3-APs and 4-APs
    # gains_3[i] = list of (y, z) such that {i, y, z} is a 3-AP
    # blocks_4[i] = list of (y, z, w) such that {i, y, z, w} is a 4-AP
    
    gains_3 = [[] for _ in range(N + 1)]
    blocks_4 = [[] for _ in range(N + 1)]
    
    # 3-APs
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            gains_3[a].append((b, c))
            gains_3[b].append((a, c))
            gains_3[c].append((a, b))
            
    # 4-APs
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            d4 = a + 3 * d
            blocks_4[a].append((b, c, d4))
            blocks_4[b].append((a, c, d4))
            blocks_4[c].append((a, b, d4))
            blocks_4[d4].append((a, b, c))
            
    def can_add(x, A, exclude=None):
        """Check if adding x to A is valid (no 4-APs).
        If exclude is not None, treat exclude as not in A for the check."""
        for y, z, w in blocks_4[x]:
            # Check if y, z, w are all in A (excluding 'exclude')
            y_in = A[y] and (y != exclude)
            z_in = A[z] and (z != exclude)
            w_in = A[w] and (w != exclude)
            if y_in and z_in and w_in:
                return False
        return True
        
    def calc_gain_add(x, A, exclude=None):
        """Calculate increase in 3-AP count if x is added to A (excluding 'exclude')."""
        gain = 0
        for y, z in gains_3[x]:
            y_in = A[y] and (y != exclude)
            z_in = A[z] and (z != exclude)
            if y_in and z_in:
                gain += 1
        return gain
        
    def calc_loss_remove(x, A):
        """Calculate decrease in 3-AP count if x is removed from A."""
        loss = 0
        for y, z in gains_3[x]:
            if A[y] and A[z]:
                loss += 1
        return loss

    def count_3aps(A):
        """Exact count of 3-APs in A."""
        count = 0
        # Iterate all 3-APs
        for d in range(1, N // 2 + 1):
            for a in range(1, N - 2 * d + 1):
                b = a + d
                c = a + 2 * d
                if A[a] and A[b] and A[c]:
                    count += 1
        return count

    def cantor_set():
        A = [0] * (N + 1)
        for i in range(1, N + 1):
            n = i
            valid = True
            while n > 0:
                if n % 3 == 1:
                    valid = False
                    break
                n //= 3
            if valid:
                A[i] = 1
        return A

    A = cantor_set()
    current_score = count_3aps(A)
    best_score = current_score
    best_A = A[:]
    
    # Simulated Annealing parameters
    T = 10.0
    cooling = 0.9999
    min_T = 0.5
    
    steps = 0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if T < min_T:
            # Restart
            T = 10.0
            A = best_A[:]
            # Small perturbation
            for _ in range(5):
                i = random.randint(1, N)
                if A[i] == 0:
                    if can_add(i, A):
                        A[i] = 1
                else:
                    A[i] = 0
            current_score = count_3aps(A)
            no_improve = 0
            
        # Random move: Flip or Swap
        move_type = random.random()
        
        if move_type < 0.5:
            # Flip
            i = random.randint(1, N)
            if A[i] == 0:
                if can_add(i, A):
                    delta = calc_gain_add(i, A)
                    if delta > 0 or random.random() < (delta / T):
                        A[i] = 1
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_A = A[:]
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                delta = -calc_loss_remove(i, A)
                if delta > 0 or random.random() < (delta / T):
                    A[i] = 0
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_A = A[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
        else:
            # Swap
            i = random.randint(1, N)
            j = random.randint(1, N)
            if i == j:
                continue
            if A[i] == 1 and A[j] == 0:
                # Try swapping i out, j in
                if can_add(j, A, exclude=i):
                    loss = calc_loss_remove(i, A)
                    gain = calc_gain_add(j, A, exclude=i)
                    delta = gain - loss
                    if delta > 0 or random.random() < (delta / T):
                        A[i] = 0
                        A[j] = 1
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_A = A[:]
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        T *= cooling
        if no_improve > max_no_improve:
            T = 10.0
            no_improve = 0

    # Ensure we return a valid set
    # best_A is always valid by construction (we only accept valid moves)
    # But just in case, verify
    # (Verification is expensive, skip if confident)
    
    return [i for i in range(1, N + 1) if best_A[i]]
