import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety
    
    def encode(triple):
        a, b, c = triple
        return a * v * v + b * v + c
    
    def decode(code):
        c = code % v
        b = (code // v) % v
        a = code // (v * v)
        return (a, b, c)

    all_triples = set(encode(combo) for combo in itertools.combinations(range(v), t))
    
    best_blocks = None
    best_count = float('inf')
    
    max_restarts = 100 # Can run many due to speed
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        uncovered = set(all_triples)
        blocks = []
        
        while uncovered:
            # Pick target triple
            if restart == 0:
                target_triple = min(uncovered)
            else:
                # Efficient random pick
                # list(uncovered) is O(N). N <= 2024.
                target_triple = random.choice(list(uncovered))
            
            t_list = decode(target_triple)
            block = list(t_list)
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                best_rand = -1.0
                
                # Pre-compute pairs in block
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    for pair in pairs_in_block:
                        triple_elements = sorted([x, pair[0], pair[1]])
                        code = encode(triple_elements)
                        if code in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        if restart > 0:
                            best_rand = random.random()
                    elif gain == best_gain and gain > 0:
                        if restart > 0:
                            r = random.random()
                            if r > best_rand:
                                best_x = x
                                best_rand = r
                        else:
                            if x < best_x:
                                best_x = x
            
            if best_x != -1:
                block.append(best_x)
            
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for triple in itertools.combinations(block, t):
            code = encode(triple)
            uncovered.discard(code)
        
        # Redundancy Removal
        block_sets = [set(b) for b in blocks]
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                is_redundant = True
                for triple in itertools.combinations(blocks[i], t):
                    found = False
                    for j in range(len(blocks)):
                        if i == j:
                            continue
                        if all(x in block_sets[j] for x in triple):
                            found = True
                            break
                    if not found:
                        is_redundant = False
                        break
                
                if is_redundant:
                    blocks.pop(i)
                    block_sets.pop(i)
                    changed = True
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = [b[:] for b in blocks]
                    break
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [b[:] for b in blocks]
        
        if time.time() > deadline - 10.0:
            break

    if best_blocks is None:
        return []
        
    return best_blocks
