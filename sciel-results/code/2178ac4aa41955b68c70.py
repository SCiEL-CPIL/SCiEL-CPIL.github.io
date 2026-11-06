import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    n = 256
    rng = random.Random(12345)
    
    def max_disc(s):
        max_d = 0
        for d in range(1, n // 2 + 1):
            for r in range(d):
                cur_max = 0
                cur_min = 0
                g_max = 0
                g_min = 0
                # Local variable for speed
                s_loc = s
                for i in range(r, n, d):
                    x = s_loc[i]
                    # Max subarray
                    if cur_max < 0:
                        cur_max = x
                    else:
                        cur_max += x
                    if cur_max > g_max:
                        g_max = cur_max
                        if g_max > max_d:
                            max_d = g_max
                            
                    # Min subarray
                    if cur_min > 0:
                        cur_min = x
                    else:
                        cur_min += x
                    if cur_min < g_min:
                        g_min = cur_min
                        neg_g_min = -g_min
                        if neg_g_min > max_d:
                            max_d = neg_g_min
        return max_d

    # Initial
    s = [rng.choice((-1, 1)) for _ in range(n)]
    curr_d = max_disc(s)
    best_s = s.copy()
    best_d = curr_d
    
    while time.time() < deadline:
        indices = list(range(n))
        rng.shuffle(indices)
        
        improved = False
        for i in indices:
            if time.time() > deadline:
                return best_s
            
            s[i] *= -1
            new_d = max_disc(s)
            
            if new_d < curr_d:
                curr_d = new_d
                improved = True
                if curr_d < best_d:
                    best_d = curr_d
                    best_s = s.copy()
            else:
                s[i] *= -1
        
        if not improved:
            # Restart
            s = [rng.choice((-1, 1)) for _ in range(n)]
            curr_d = max_disc(s)
            if curr_d < best_d:
                best_d = curr_d
                best_s = s.copy()
            
            if time.time() > deadline:
                break

    return best_s
