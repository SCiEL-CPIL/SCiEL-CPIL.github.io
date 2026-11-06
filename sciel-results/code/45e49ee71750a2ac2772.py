import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Precompute shift masks for fast scoring
    # For a given bitset b, the score is the number of d in 1..255
    # such that (b & (b >> d) & (b >> 2*d)) != 0
    
    def score_bitset(b):
        count = 0
        for d in range(1, 256):
            if (b & (b >> d) & (b >> (2 * d))):
                count += 1
        return count
    
    def get_bitset(S):
        b = 0
        for x in S:
            b |= (1 << (x - 1))
        return b
    
    def random_set():
        return random.sample(range(1, 513), 32)
    
    # Try structured initializations that may have good properties
    def structured_init():
        """Create sets with arithmetic structure to seed good search regions."""
        r = random.random()
        if r < 0.3:
            # Random arithmetic progression base + noise
            base = random.randint(1, 10)
            step = random.randint(1, 15)
            vals = set()
            for i in range(32):
                v = base + i * step + random.randint(-3, 3)
                v = max(1, min(512, v))
                vals.add(v)
            while len(vals) < 32:
                vals.add(random.randint(1, 512))
            return list(vals)[:32]
        elif r < 0.5:
            # Two interleaved APs
            base1 = random.randint(1, 50)
            base2 = random.randint(1, 50)
            step = random.randint(8, 20)
            vals = set()
            for i in range(16):
                vals.add(base1 + i * step)
            for i in range(16):
                vals.add(base2 + i * step)
            # Deduplicate and fill
            result = []
            for v in sorted(vals):
                if 1 <= v <= 512 and v not in result:
                    result.append(v)
            while len(result) < 32:
                result.append(random.randint(1, 512))
            return result[:32]
        else:
            return random_set()
    
    best_set = None
    best_bit = None
    best_score = -1
    
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() >= deadline - 15:
            break
        
        current_set = structured_init()
        current_bit = get_bitset(current_set)
        current_score = score_bitset(current_bit)
        
        # Simulated Annealing with adaptive parameters
        T = 20.0
        T_min = 0.02
        stall_count = 0
        stall_threshold = 1500
        no_improve_steps = 0
        local_best_score = current_score
        
        while time.time() < deadline - 15 and T > T_min:
            # Generate neighbor: replace one element
            idx = random.randint(0, 31)
            old_val = current_set[idx]
            new_val = random.randint(1, 512)
            
            if new_val == old_val or new_val in current_set:
                continue
            
            # Update bitset
            new_bit = current_bit
            new_bit &= ~(1 << (old_val - 1))
            new_bit |= (1 << (new_val - 1))
            
            new_score = score_bitset(new_bit)
            delta = new_score - current_score
            
            # Acceptance: Metropolis criterion
            if delta >= 0:
                accept = True
            else:
                # exp(delta/T) with early exit for very negative delta
                if delta < -10 * T:
                    accept = False
                else:
                    accept = random.random() < (2.718281828 ** (delta / T))
            
            if accept:
                current_set[idx] = new_val
                current_bit = new_bit
                current_score = new_score
                stall_count = 0
                
                if current_score > local_best_score:
                    local_best_score = current_score
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                    
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
                    best_bit = current_bit
                    
            else:
                stall_count += 1
                no_improve_steps += 1
            
            # Adaptive cooling: cool faster when stuck
            if stall_count > stall_threshold:
                T *= 0.9
                stall_count = 0
            
            # If no improvement for a long time, kick temperature up slightly
            if no_improve_steps > 5000 and T > T_min * 10:
                T = max(T, 5.0)
                no_improve_steps = 0
            
            T *= 0.99997
        
        # Final polish: exhaustive single-element replacement, multi-pass
        if time.time() < deadline - 5:
            for polish_pass in range(3):
                if time.time() >= deadline - 5:
                    break
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    for i in range(32):
                        if time.time() >= deadline - 5:
                            break
                        old_val = current_set[i]
                        best_new_val = None
                        best_new_score = current_score
                        for new_val in range(1, 513):
                            if new_val == old_val or new_val in current_set:
                                continue
                            new_bit = current_bit
                            new_bit &= ~(1 << (old_val - 1))
                            new_bit |= (1 << (new_val - 1))
                            new_score = score_bitset(new_bit)
                            if new_score > best_new_score:
                                best_new_score = new_score
                                best_new_val = new_val
                        if best_new_val is not None:
                            current_set[i] = best_new_val
                            current_bit = get_bitset(current_set)
                            current_score = best_new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(current_set)
                                best_bit = current_bit
    
    if best_set is None:
        best_set = list(range(1, 33))
        
    return best_set
