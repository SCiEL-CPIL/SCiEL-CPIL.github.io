import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: aim to finish by 115s to be well within the 120s kill switch
    deadline = start_time + 115.0
    
    total_words = q ** n
    
    # Helper functions for conversion
    def int_to_tuple(x, n, q):
        t = [0] * n
        for i in range(n):
            t[i] = x % q
            x //= q
        return t

    def tuple_to_int(t, q):
        x = 0
        for i in range(n-1, -1, -1):
            x = x * q + t[i]
        return x

    # Precompute balls (neighbors within radius)
    # For radius=1, ball size is 1 + n*(q-1)
    balls = [None] * total_words
    for i in range(total_words):
        t_i = int_to_tuple(i, n, q)
        ball = [i]
        if radius >= 1:
            for pos in range(n):
                orig_val = t_i[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    t_new = t_i[:]
                    t_new[pos] = new_val
                    ball.append(tuple_to_int(t_new, q))
        balls[i] = ball
        
    # Coverage matrix M: M[i, j] = 1 if j is in balls[i]
    # This allows fast coverage checks: M[codeword] gives the mask of covered words
    M = np.zeros((total_words, total_words), dtype=np.uint8)
    for i in range(total_words):
        M[i, balls[i]] = 1

    def is_valid(codewords):
        if len(codewords) == 0:
            return total_words == 0
        # Vectorized check: sum of coverage for each word must be >= 1
        # M[codewords, :] is (K, N). Sum over axis 0 gives (N,)
        coverage_counts = M[codewords, :].sum(axis=0)
        return np.all(coverage_counts >= 1)

    def greedy_cover(initial_codewords=None):
        """
        Greedy set cover. 
        If initial_codewords is provided, it starts with those and adds more if needed.
        """
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        
        # Start with initial codewords if provided
        if initial_codewords:
            # Ensure no duplicates and valid indices
            seen = set()
            for c in initial_codewords:
                if 0 <= c < total_words and c not in seen:
                    codewords.append(c)
                    seen.add(c)
                    covered[balls[c]] = True
        
        while True:
            if time.time() > deadline - 5:
                break
            uncovered_mask = ~covered
            if not np.any(uncovered_mask):
                break
            
            # Scores: number of currently uncovered words covered by each candidate
            # We can optimize by only considering candidates not already in codewords,
            # but M.dot is fast enough for N=2187.
            scores = M.dot(uncovered_mask.astype(np.uint8))
            
            # Mask out already selected codewords to avoid re-adding
            # Actually, adding a duplicate doesn't help coverage, but M.dot will give 0 
            # for covered parts anyway. But to be safe, we can just pick argmax.
            # If a codeword is already in, its contribution to *new* coverage is 0 
            # because its ball is already covered? 
            # Wait, if a codeword is in, its ball is covered. So uncovered_mask is 0 there.
            # So scores[existing] will be 0. So argmax will naturally pick new ones 
            # unless all scores are 0 (which means we are done or stuck).
            
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                break
                
            codewords.append(best_idx)
            covered[balls[best_idx]] = True
            
        return codewords

    def local_search_remove(codewords):
        """
        Iteratively remove redundant codewords.
        A codeword is redundant if removing it does not uncover any word.
        """
        if not codewords:
            return codewords
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            # Compute coverage counts
            # M[codewords, :] is (K, N). Sum over axis 0.
            cover_count = M[codewords, :].sum(axis=0)
            
            # Identify redundant indices
            for i in range(len(codewords)):
                c = codewords[i]
                # Check if removing c leaves all words in balls[c] covered by others
                # i.e., cover_count[word] > 1 for all word in balls[c]
                if np.min(cover_count[balls[c]]) > 1:
                    codewords.pop(i)
                    changed = True
                    break # Restart loop to update cover_count
            
        return codewords

    def generate_linear_code_seed():
        """
        Generate a random linear code of dimension k over GF(q).
        For q=3, n=7, let's try k=4 or k=5.
        k=4 -> 3^4 = 81 codewords.
        k=5 -> 3^5 = 243 codewords (might be too large to start, but covers more).
        Let's stick to k=4 for a "sparse" seed that requires greedy extension.
        """
        k = 4
        # Generate random generator matrix G (k x n) over GF(q)
        G = np.random.randint(0, q, size=(k, n), dtype=np.int64)
        
        # Ensure rank is k? 
        # For small k, random matrices are likely full rank. 
        # If not, the code size is smaller, which is fine.
        
        # Generate all codewords: c = u * G mod q
        # u ranges over all q^k vectors
        codewords = []
        # Iterate over all u
        # We can do this efficiently by iterating 0 to q^k - 1 and converting to base q
        num_codewords = q ** k
        for u_int in range(num_codewords):
            u = int_to_tuple(u_int, k, q)
            # c = sum(u_i * G_i) mod q
            # c is a vector of length n
            c_vec = np.zeros(n, dtype=np.int64)
            for i in range(k):
                c_vec += u[i] * G[i]
            c_vec = c_vec % q
            c_int = tuple_to_int(c_vec, q)
            codewords.append(c_int)
            
        return codewords

    # Main Strategy: Multi-Start with Algebraic Seeds
    # 1. Try a few different random linear code seeds
    # 2. For each, run greedy cover starting from the seed
    # 3. Run local search
    # 4. Keep the best
    
    best_codewords = []
    best_size = float('inf')
    
    # How many seeds to try?
    # Each iteration is fast. 
    # Greedy is O(N^2) roughly? No, O(N * (1 + n(q-1))) per step, N steps.
    # N=2187. Ball size = 15. 
    # M.dot is 2187x2187 matrix vector mult. ~4M ops. 
    # 2187 steps * 4M ops = ~10^10 ops? No, numpy is fast. 
    # Let's estimate: 2187 iterations. Each M.dot is fast. 
    # Local search is also fast.
    # We can probably do 20-50 seeds in 110s.
    
    num_seeds = 30
    
    for i in range(num_seeds):
        if time.time() > deadline - 10:
            break
            
        # Generate a random linear code seed
        seed_codewords = generate_linear_code_seed()
        
        # Greedy cover starting from seed
        # Note: The seed might not be a valid cover. Greedy will add necessary codewords.
        current = greedy_cover(seed_codewords)
        
        # Local search
        current = local_search_remove(current)
        
        if len(current) < best_size:
            best_size = len(current)
            best_codewords = current[:]
            
        # If we hit a very good size, we might want to stop early?
        # Best known is 186. If we hit < 190, we are doing great.
        # But let's keep going to see if we can do better.
        
    # Final check: if best_codewords is empty (shouldn't happen), fallback to pure greedy
    if not best_codewords:
        best_codewords = greedy_cover()
        best_codewords = local_search_remove(best_codewords)

    # Convert to output format
    result = []
    for c in best_codewords:
        result.append(int_to_tuple(c, n, q))
        
    return result
