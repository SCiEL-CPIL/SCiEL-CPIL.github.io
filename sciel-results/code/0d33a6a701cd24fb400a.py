import random

def construct():
       # 1. Generate Seed
       seed_words = []
       for slope in range(16):
           for intercept in range(2):
               word = 0
               for point in range(16):
                   bit = ((slope & point).bit_count() + intercept) % 2
                   word |= bit << point
               seed_words.append(word)
       
       # 2. Generate Pool
       # Candidates: even weight, weight >= 6, bit 16 is 0 (represented by 16-bit int)
       pool = []
       for w in range(1 << 16):
           bc = w.bit_count()
           if bc % 2 == 0 and bc >= 6:
               pool.append(w)
       
       # Remove seed words from pool
       seed_set = set(seed_words)
       pool = [w for w in pool if w not in seed_set]
       
       best_code = seed_words[:]
       
       # 3. Greedy Search
       # To ensure reproducibility or just random, we use random.shuffle
       for _ in range(20):
           random.shuffle(pool)
           current_code = seed_words[:]
           code_set = set(seed_words) # For fast membership check? Not needed for distance.
           
           for w in pool:
               # Check distance to all in current_code
               valid = True
               for c in current_code:
                   if (w ^ c).bit_count() < 6:
                       valid = False
                       break
               if valid:
                   current_code.append(w)
                   
           if len(current_code) > len(best_code):
               best_code = current_code[:]
               # Early stop if we hit a high target?
               if len(best_code) > 300:
                   break
                   
       return best_code
