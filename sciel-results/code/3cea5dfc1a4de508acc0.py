import time

def construct():
        t_start = time.time()
        deadline = t_start + 80.0 # Safety margin
        N = 4096
        MAX_SIZE = 80
        
        selected = []
        sums = set()
        
        for x in range(1, N + 1):
            if len(selected) >= MAX_SIZE:
                break
            if time.time() > deadline:
                break
            
            # Generate new sums involving x
            new_sums = set()
            # 1. x+x+x
            new_sums.add(3*x)
            
            # 2. x+x+a
            for a in selected:
                s = 2*x + a
                if s in new_sums:
                    break # Internal collision
                new_sums.add(s)
            else:
                # 3. x+a+b
                for i, a in enumerate(selected):
                    for b in selected[i:]:
                        s = x + a + b
                        if s in new_sums:
                            break
                        new_sums.add(s)
                    if len(new_sums) != 1 + len(selected) + (len(selected)*(len(selected)+1)//2):
                         # This check is expensive. Better to just let set handle duplicates?
                         # If we add to set, duplicates are ignored. 
                         # We need to detect if *two different triples* produce the same sum.
                         # If we just use a set, we lose the count.
                         # So we must check for duplicates explicitly or use a list and sort?
                         # No, checking `if s in new_sums` before adding is O(1) on average.
                         # If we find a duplicate, we reject x.
                         pass
                
                # If we broke out of inner loop due to collision, reject
                # We need a flag
                valid = True
                new_sums = set()
                new_sums.add(3*x)
                
                for a in selected:
                    s = 2*x + a
                    if s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
                
                if valid:
                    for i, a in enumerate(selected):
                        for b in selected[i:]:
                            s = x + a + b
                            if s in new_sums:
                                valid = False
                                break
                            new_sums.add(s)
                        if not valid:
                            break
                
                if valid and not (new_sums & sums):
                    selected.append(x)
                    sums.update(new_sums)
                    
        return selected
