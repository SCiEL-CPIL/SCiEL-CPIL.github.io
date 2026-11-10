import time

def construct():
        start_time = time.time()
        deadline = start_time + 50 # 50s margin
        
        numbers = list(range(2, 129))
        n = len(numbers)
        
        best_seq = []
        best_len = 0
        
        # Initial Greedy to get a good LB
        def greedy():
            vals = []
            prods = set()
            for x in numbers:
                tails, prod = [x], x
                for a in reversed(vals):
                    prod *= a
                    tails.append(prod)
                if not prods.intersection(tails):
                    vals.append(x)
                    prods.update(tails)
            return vals
        
        init_seq = greedy()
        best_len = len(init_seq)
        best_seq = init_seq
        
        # DFS
        # State: idx (current number to consider), current_seq, current_prods
        # We pass lists to avoid copying, or use tuples?
        # Sets are mutable, so we need to undo changes (backtrack).
        
        def dfs(idx, seq, prods):
            nonlocal best_len, best_seq, deadline
            
            if time.time() > deadline:
                return
                
            # Pruning
            if len(seq) + (n - idx) <= best_len:
                return
                
            if idx == n:
                if len(seq) > best_len:
                    best_len = len(seq)
                    best_seq = seq[:]
                return
                
            x = numbers[idx]
            
            # Option 1: Include x
            # Check validity
            tails = [x]
            prod = x
            for a in reversed(seq):
                prod *= a
                tails.append(prod)
                
            # Check conflict
            conflict = False
            for p in tails:
                if p in prods:
                    conflict = True
                    break
            
            if not conflict:
                # Add
                seq.append(x)
                for p in tails:
                    prods.add(p)
                    
                dfs(idx + 1, seq, prods)
                
                # Undo
                for p in tails:
                    prods.remove(p)
                seq.pop()
            
            # Option 2: Exclude x
            dfs(idx + 1, seq, prods)
            
        dfs(0, [], set())
        
        return best_seq
