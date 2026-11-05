import numpy as np
import time

def construct():
        N = 17
        D = 6
        seed_size = 32
        deadline = time.time() + 60.0
        
        # Start with a random code
        # Or try to construct the 256-word code
        # For now, start with a random code of size 50
        words = []
        for _ in range(50):
            w = np.random.randint(0, 2**N)
            # Check distance
            valid = True
            for x in words:
                if np.bitwise_xor(w, x).bit_count() < D:
                    valid = False
                    break
            if valid:
                words.append(w)
                
        # Local search
        best_words = words[:]
        while time.time() < deadline - 5.0:
            # Try to add a new word
            added = False
            for _ in range(1000):
                w = np.random.randint(0, 2**N)
                valid = True
                for x in best_words:
                    if np.bitwise_xor(w, x).bit_count() < D:
                        valid = False
                        break
                if valid:
                    best_words.append(w)
                    added = True
                    break
            if not added:
                # Try to swap
                # ...
                pass
                
        return [int(w) for w in best_words]
