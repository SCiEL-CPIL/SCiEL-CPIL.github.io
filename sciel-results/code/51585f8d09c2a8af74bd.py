import random
import time

def _count_k222(masks, n):
    total = 0
    for u in range(n):
        mu = masks[u]
        for v in range(u + 1, n):
            if (mu >> v) & 1:
                s = mu & masks[v]
                s &= ~(1 << u)
                s &= ~(1 << v)
                
                x_iter = s
                while x_iter:
                    lsb_x = x_iter & -x_iter
                    x = lsb_x.bit_length() - 1
                    x_iter ^= lsb_x
                    
                    s_without_x = s & ~lsb_x
                    y_iter = s_without_x
                    while y_iter:
                        lsb_y = y_iter & -y_iter
                        y = lsb_y.bit_length() - 1
                        y_iter ^= lsb_y
                        
                        common = masks[x] & masks[y] & s
                        k = common.bit_count()
                        if k >= 2:
                            total += k * (k - 1) // 2
    return total // 6

def _build_masks_paley(n):
    # n must be prime, n % 4 == 1
    squares = {x * x % n for x in range(1, n)}
    masks = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in squares:
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)
    return masks

def _masks_to_colors(masks, n):
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(1 if (masks[i] >> j) & 1 else 0)
    return colors

def _sa_search(n, time_limit):
    start_time = time.time()
    best_energy = float('inf')
    best_masks = None
    
    for restart in range(5):
        if time.time() - start_time > time_limit:
            break
            
        masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
        
        energy = _count_k222(masks, n)
        if energy == 0:
            return masks
        
        T = 10.0
        cooling = 0.999
        current_masks = masks[:]
        current_energy = energy
        
        while time.time() - start_time < time_limit:
            if current_energy == 0:
                return current_masks
            
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            bit = 1 << j
            if current_masks[i] & bit:
                current_masks[i] ^= bit
                current_masks[j] ^= (1 << i)
            else:
                current_masks[i] |= bit
                current_masks[j] |= (1 << i)
            
            new_energy = _count_k222(current_masks, n)
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_masks = current_masks[:]
            else:
                if current_masks[i] & bit:
                    current_masks[i] ^= bit
                    current_masks[j] ^= (1 << i)
                else:
                    current_masks[i] |= bit
                    current_masks[j] |= (1 << i)
            
            T *= cooling
            if T < 0.01:
                break
        
        if best_energy == 0:
            return best_masks
            
    return best_masks if best_energy == 0 else None

def construct():
    deadline = time.time() + 110.0
    
    # Try Paley 29
    try:
        masks29 = _build_masks_paley(29)
        e29 = _count_k222(masks29, 29)
        if e29 == 0:
            colors = _masks_to_colors(masks29, 29)
            return {'n': 29, 'colors': colors}
    except Exception:
        pass
        
    # Try SA for 30
    if time.time() < deadline:
        masks = _sa_search(30, time.time() + 40.0)
        if masks:
            colors = _masks_to_colors(masks, 30)
            return {'n': 30, 'colors': colors}
            
    # Fallback to Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    colors = [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]
    return {'n': 17, 'colors': colors}
