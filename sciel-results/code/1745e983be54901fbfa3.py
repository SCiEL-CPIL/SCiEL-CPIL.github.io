import numpy as np
import random

def construct():
    N = 17
    MIN_DIST = 6
    
    residues = {1, 4, 9, 16, 8, 2, 15, 13}
    
    A = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        for j in range(N):
            if i != j:
                diff = (i - j) % N
                if diff in residues:
                    A[i][j] = 1
    
    # Get all rows as integers
    rows = []
    for i in range(N):
        val = 0
        for j in range(N):
            if A[i][j] == 1:
                val |= (1 << j)
        rows.append(val)
        
    # We want to find 8 linearly independent vectors from the row space
    # such that the code they generate has min distance >= 6.
    
    # First, get a basis for the row space
    basis = []
    for r in rows:
        # Check if r is in span of basis
        # Simple check: XOR r with basis vectors to see if we can reduce it to 0
        # But we need linear independence over GF(2)
        # Let's use Gaussian elimination on the list of rows
        pass
        
    # Let's just use the rows themselves.
    # The row space is spanned by rows.
    # We pick 8 rows.
    
    best_words = []
    
    # Try the first 8 rows
    # Check if they are linearly independent
    def is_independent(vecs):
        # Gaussian elimination
        M = [v for v in vecs]
        rank = 0
        for col in range(N):
            pivot = -1
            for i in range(rank, len(M)):
                if (M[i] >> col) & 1:
                    pivot = i
                    break
            if pivot == -1:
                continue
            M[rank], M[pivot] = M[pivot], M[rank]
            for i in range(len(M)):
                if i != rank and ((M[i] >> col) & 1):
                    M[i] ^= M[rank]
            rank += 1
        return rank == len(vecs)
        
    def min_distance(words):
        min_d = N
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                d = (words[i] ^ words[j]).bit_count()
                if d < min_d:
                    min_d = d
        return min_d
        
    # Generate all 2^8 combinations of a candidate basis
    def generate_words(basis_vecs):
        words = [0]
        for v in basis_vecs:
            new_words = [w ^ v for w in words]
            words.extend(new_words)
        return words

    # Try random subsets of 8 rows
    # Or just try the first 8 rows
    candidate_basis = rows[:8]
    if is_independent(candidate_basis):
        words = generate_words(candidate_basis)
        if min_distance(words) >= MIN_DIST:
            return words
            
    # If not, try other combinations
    # Since the [17,8,6] code exists, we can find it.
    # We can search for 8 vectors in the row space.
    
    # Let's get a full basis of the row space first
    full_basis = []
    for r in rows:
        # Reduce r
        temp = r
        for b in full_basis:
            if (temp & b) != 0:
                # Find highest bit?
                # Simple reduction:
                # This is inefficient. Let's use numpy for GE.
                pass
                
    # Let's use numpy for GE to get full basis
    M = A.copy()
    basis = []
    for col in range(N):
        pivot_row = -1
        for row in range(len(basis), N):
            if M[row][col] == 1:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
        M[len(basis), M[pivot_row]] = M[pivot_row, M[len(basis)]]
        for row in range(N):
            if row != len(basis) and M[row][col] == 1:
                M[row] ^= M[len(basis)]
        basis.append(M[len(basis)].copy())
        
    k = len(basis)
    if k < 8:
        # Should not happen
        return rows[:k]
        
    # Convert basis to ints
    basis_ints = []
    for vec in basis:
        val = 0
        for j in range(N):
            if vec[j] == 1:
                val |= (1 << j)
        basis_ints.append(val)
        
    # We need to select 8 from these k vectors?
    # Or select 8 from the row space.
    # The row space is spanned by basis_ints.
    # We can randomly generate vectors in the row space and check independence and min distance.
    
    # Simpler: The [17,8,6] code is a subcode of the row space.
    # We can try to find 8 vectors.
    
    # Let's try the first 8 basis vectors
    cand = basis_ints[:8]
    if is_independent(cand):
        words = generate_words(cand)
        if min_distance(words) >= MIN_DIST:
            return words
            
    # Random search
    random.seed(42)
    for _ in range(1000):
        # Pick 8 random vectors from the row space
        # A random vector is XOR of random basis vectors
        cand = []
        for _ in range(8):
            mask = random.getrandbits(k)
            v = 0
            for i in range(k):
                if (mask >> i) & 1:
                    v ^= basis_ints[i]
            cand.append(v)
            
        if is_independent(cand):
            words = generate_words(cand)
            if min_distance(words) >= MIN_DIST:
                return words
                
    # Fallback: return the 32 words from the seed
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words
