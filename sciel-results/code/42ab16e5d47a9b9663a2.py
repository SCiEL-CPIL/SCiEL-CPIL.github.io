import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit

    # Global state for the current sequence
    seq = []
    product_set = set()

    def reset_state(new_seq):
        nonlocal seq, product_set
        seq = new_seq[:]
        product_set = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                product_set.add(p)

    def can_insert(pos, x):
        """Check if inserting x at pos creates any new product conflicts."""
        n = len(seq)
        
        # Generate suffix products ending at pos-1 (including x)
        # suf[0] = x
        # suf[1] = x * seq[pos-1]
        # ...
        suf = [x]
        p = x
        for i in range(pos - 1, -1, -1):
            p *= seq[i]
            suf.append(p)
            
        # Check left extensions (suf) against product_set
        for p in suf:
            if p in product_set:
                return False

        # Generate prefix products starting at pos
        # pre[0] = 1
        # pre[1] = seq[pos]
        # ...
        pre = [1]
        p = 1
        for j in range(pos, n):
            p *= seq[j]
            pre.append(p)
            
        # Check right extensions (x * pre[j]) against product_set
        for j in range(1, len(pre)):
            p = x * pre[j]
            if p in product_set:
                return False

        # Check middle extensions (suf[i] * pre[j]) for i > 0, j > 0
        for i in range(1, len(suf)):
            for j in range(1, len(pre)):
                p = suf[i] * pre[j]
                if p in product_set:
                    return False
                    # No need to add to new set, we just need to know if it's valid
            
        return True

    def do_insert(pos, x):
        """Insert x at pos and update product_set."""
        nonlocal seq, product_set
        n = len(seq)
        
        # Calculate new products
        suf = [x]
        p = x
        for i in range(pos - 1, -1, -1):
            p *= seq[i]
            suf.append(p)
            
        pre = [1]
        p = 1
        for j in range(pos, n):
            p *= seq[j]
            pre.append(p)
            
        # Add left extensions
        for p in suf:
            product_set.add(p)
            
        # Add right extensions
        for j in range(1, len(pre)):
            product_set.add(x * pre[j])
            
        # Add middle extensions
        for i in range(1, len(suf)):
            for j in range(1, len(pre)):
                product_set.add(suf[i] * pre[j])
                
        # Update sequence
        seq.insert(pos, x)

    def remove_element(idx):
        """Remove element at idx and update product_set."""
        nonlocal seq, product_set
        # This is harder to update incrementally.
        # Given the small size, we can just rebuild product_set.
        # Or we can remove products involving seq[idx].
        # Rebuilding is O(N^2) which is fine for N=50.
        x = seq[idx]
        seq.pop(idx)
        reset_state(seq)

    def greedy_construct(strategy='standard'):
        """Construct a sequence using a greedy strategy."""
        seq = []
        product_set = set()
        candidates = list(range(2, 129))
        
        if strategy == 'random':
            random.shuffle(candidates)
        elif strategy == 'primes':
            # Prioritize primes
            def is_prime(n):
                if n < 2: return False
                for i in range(2, int(n**0.5)+1):
                    if n % i == 0: return False
                return True
            primes = [x for x in candidates if is_prime(x)]
            composites = [x for x in candidates if not is_prime(x)]
            candidates = primes + composites
            random.shuffle(primes) # Shuffle primes to add diversity
            random.shuffle(composites)
            candidates = primes + composites

        for x in candidates:
            # Find insertion position
            pos = 0
            for i, v in enumerate(seq):
                if v < x:
                    pos = i + 1
                else:
                    break
            
            # Check if valid
            # We need to check against the current product_set
            # But can_insert uses the global 'seq' and 'product_set'.
            # So we need to temporarily set them or pass them.
            # Let's refactor can_insert to take seq and product_set as args.
            
            # For now, let's just use a simple check function here.
            # Actually, let's just use the global state for the main search,
            # and for greedy, we can use a local version.
            
            # To avoid code duplication, let's just use the global state
            # by resetting it before each greedy run.
            pass # Placeholder, will implement below

    # Refactored Greedy
    def run_greedy(strategy='standard'):
        local_seq = []
        local_prod_set = set()
        candidates = list(range(2, 129))
        
        if strategy == 'random':
            random.shuffle(candidates)
        elif strategy == 'primes':
            def is_prime(n):
                if n < 2: return False
                for i in range(2, int(n**0.5)+1):
                    if n % i == 0: return False
                return True
            primes = [x for x in candidates if is_prime(x)]
            composites = [x for x in candidates if not is_prime(x)]
            random.shuffle(primes)
            random.shuffle(composites)
            candidates = primes + composites

        for x in candidates:
            pos = 0
            for i, v in enumerate(local_seq):
                if v < x:
                    pos = i + 1
                else:
                    break
            
            # Check validity
            valid = True
            n = len(local_seq)
            
            # Left extensions
            p = x
            if p in local_prod_set: valid = False
            for i in range(pos-1, -1, -1):
                p *= local_seq[i]
                if p in local_prod_set: valid = False
            
            if valid:
                # Right extensions
                p = x
                for j in range(pos, n):
                    p *= local_seq[j]
                    if p in local_prod_set: valid = False
            
            if valid:
                # Middle extensions
                suf = [x]
                p = x
                for i in range(pos-1, -1, -1):
                    p *= local_seq[i]
                    suf.append(p)
                    
                pre = [1]
                p = 1
                for j in range(pos, n):
                    p *= local_seq[j]
                    pre.append(p)
                    
                for i in range(1, len(suf)):
                    for j in range(1, len(pre)):
                        if suf[i] * pre[j] in local_prod_set:
                            valid = False
                            break
                    if not valid: break
            
            if valid:
                # Add to local_seq and local_prod_set
                local_seq.insert(pos, x)
                # Update local_prod_set
                # Reuse the logic from do_insert but for local vars
                # For efficiency, we can just rebuild local_prod_set if N is small,
                # but incremental is better.
                
                # Let's just rebuild for simplicity in greedy, N is small
                # No, incremental is easy.
                # Left
                p = x
                local_prod_set.add(p)
                for i in range(pos-1, -1, -1):
                    p *= local_seq[i-1] # Wait, local_seq is updated? No, we insert after check.
                    # Careful with indices.
                
                # Let's just rebuild local_prod_set. It's O(N^2) and N <= 128.
                # 128^2 = 16384 operations. Very fast.
                local_prod_set.clear()
                for i in range(len(local_seq)):
                    p = 1
                    for j in range(i, len(local_seq)):
                        p *= local_seq[j]
                        local_prod_set.add(p)
                        
        return local_seq

    # Main Search Loop
    best_seq = []
    best_len = 0
    
    # Initialize with a few greedy strategies
    initial_seqs = [
        run_greedy('standard'),
        run_greedy('primes'),
        run_greedy('random'),
        run_greedy('random')
    ]
    
    best_seq = max(initial_seqs, key=len)
    best_len = len(best_seq)
    
    # Local Search (Hill Climbing with Kicks)
    current_seq = best_seq[:]
    
    def update_best(seq):
        nonlocal best_seq, best_len
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

    update_best(current_seq)
    
    while time.time() < deadline:
        # Try to add elements
        in_seq = set(current_seq)
        available = [x for x in range(2, 129) if x not in in_seq]
        
        # Shuffle available to randomize addition order
        random.shuffle(available)
        
        added_any = False
        for x in available:
            pos = 0
            for i, v in enumerate(current_seq):
                if v < x:
                    pos = i + 1
                else:
                    break
            
            # Check validity
            # We need a check function that works on current_seq
            # Let's use the global state helper by resetting it
            # This is inefficient. Let's write a standalone check.
            
            # Standalone check for current_seq
            n = len(current_seq)
            valid = True
            
            # Left
            p = x
            # We need the set of products of current_seq
            # We don't have it stored globally for current_seq in this loop structure easily.
            # Let's maintain product_set for current_seq.
            
            # This approach is getting messy. Let's stick to the global state approach
            # but manage it carefully.
            
            # Let's restart the loop structure to use global state.
            pass # Placeholder

    # Let's restructure to use global state for the main search
    # We will run multiple "sessions" of local search.
    
    for session in range(100): # Max sessions
        if time.time() > deadline:
            break
            
        # Initialize session with a random perturbation of best_seq
        # Or start from a new greedy sequence
        if session == 0:
            start_seq = best_seq[:]
        else:
            # Pick a random initial sequence
            start_seq = random.choice(initial_seqs)
            # Or perturb best_seq
            if random.random() < 0.5:
                start_seq = best_seq[:]
                # Remove 2-5 random elements
                for _ in range(random.randint(2, 5)):
                    if start_seq:
                        idx = random.randint(0, len(start_seq)-1)
                        start_seq.pop(idx)
        
        reset_state(start_seq)
        
        # Hill Climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            in_seq = set(seq)
            available = [x for x in range(2, 129) if x not in in_seq]
            random.shuffle(available)
            
            for x in available:
                pos = 0
                for i, v in enumerate(seq):
                    if v < x:
                        pos = i + 1
                    else:
                        break
                
                if can_insert(pos, x):
                    do_insert(pos, x)
                    improved = True
                    update_best(seq)
                    # Restart the loop to re-shuffle and check all available again
                    # This ensures we find the maximal set for this state
                    break # Break to restart the while loop
            
            if not improved:
                # Stuck. Try a kick.
                # Remove 1-3 random elements
                kick_size = random.randint(1, 3)
                for _ in range(kick_size):
                    if seq:
                        idx = random.randint(0, len(seq)-1)
                        remove_element(idx)
                
                # After kick, try to add again
                improved = True # Force loop to continue

    return best_seq

# Let's refine the code to be cleaner and ensure correctness.
