# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0
    best_seq = []
    best_len = 0
    
    def check_and_extend(seq, prod_set, suffixes, x):
        if x in prod_set:
            return None
        
        new_suffixes = [x]
        p = x
        for s in reversed(seq):
            p *= s
            new_suffixes.append(p)
            if p in prod_set:
                return None
        
        new_prod_set = set(prod_set)
        new_prod_set.update(new_suffixes)
        new_seq = seq + [x]
        
        return new_seq, new_prod_set, new_suffixes

    def run_beam_search(width):
        nonlocal best_seq, best_len
        beam = []
        for x in range(2, 129):
            beam.append(([x], {x}, [x]))
        
        beam.sort(key=lambda s: s[0][0])
        beam = beam[:width]
        
        for _ in range(100): # Max possible length is 127
            if time.time() > deadline:
                break
            next_beam = []
            for seq, prod_set, suffixes in beam:
                last_val = seq[-1]
                for x in range(last_val + 1, 129):
                    res = check_and_extend(seq, prod_set, suffixes, x)
                    if res:
                        next_beam.append(res)
            
            if not next_beam:
                break
            
            # Heuristic: Sort by last element (smaller is better)
            # Tie-breaker: Smaller total product (smaller sum of suffixes approx)
            next_beam.sort(key=lambda s: (s[0][-1], sum(s[2])))
            
            beam = next_beam[:width]
            
            if len(beam[0][0]) > best_len:
                best_len = len(beam[0][0])
                best_seq = list(beam[0][0])

    run_beam_search(50)
    if time.time() < deadline - 10:
        run_beam_search(200)
    if time.time() < deadline - 5:
        run_beam_search(500)
        
    return best_seq
# EVOLVE-BLOCK-END
