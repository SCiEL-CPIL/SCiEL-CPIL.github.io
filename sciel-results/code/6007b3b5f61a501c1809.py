import itertools
import time

def construct():
    n = 10
    # Map int to (a, b) in Z5 x Z2
    # v = 2a + b
    def to_group(v):
        return (v // 2, v % 2)
    
    def to_int(g):
        a, b = g
        return 2 * a + b
        
    def add_g(g1, g2):
        return ((g1[0] + g2[0]) % 5, (g1[1] + g2[1]) % 2)
        
    def sub_g(g1, g2):
        return ((g1[0] - g2[0]) % 5, (g1[1] - g2[1]) % 2)

    # We need a permutation theta of range(10)
    # such that psi(c) = theta(c) - c is a permutation.
    
    # To speed up, we can just iterate.
    # But 10! is 3.6M. In Python, this might take a few seconds.
    # Let's optimize or use a known solution if possible.
    
    # Known solution for MOLS(10):
    # We can hardcode the squares if we generate them once.
    # But generating them is safer to avoid errors.
    
    # Let's try to find one quickly.
    # We can restrict the search or just run it.
    # 3.6M iterations * (10 ops) = 36M ops. Python does ~10M ops/s. So ~3-4 seconds.
    # This is well within the 60s limit.
    
    start_time = time.time()
    
    # Optimization: 
    # The condition is that the map c -> theta(c) - c is bijective.
    # This is equivalent to theta being an orthomorphism (or complete mapping depending on definition).
    
    # Let's iterate over permutations.
    # To make it faster, we can use numpy or just standard itertools.
    # Standard itertools.permutations is slow for 10!.
    # We can use a recursive generator or just hardcode a valid theta.
    
    # Let's try to find a valid theta using a simpler search or hardcoded value.
    # If we can't find one fast, we can fall back to a single square.
    # But we know it exists.
    
    # Let's try a specific structure for theta.
    # theta(a, b) = (k*a + m*b, l*b) didn't work.
    # What if theta(a, b) = (k*a + m*b, b + n*a)?
    # Let's just search.
    
    # To avoid 10! search time, we can use the fact that we only need ONE pair.
    # We can use the squares from the paper "Mutually Orthogonal Latin Squares" or similar.
    # Actually, I can construct them using the group Z5 x Z2 with a specific theta.
    
    # Let's try theta(a, b) = (2a + b, b)
    # theta is a permutation?
    # (a, b) -> (2a+b, b).
    # Inverse: b' = b, a' = (2a+b-b)/2? No, 2 is invertible mod 5.
    # a = (a' - b') * 3 (since 2*3=6=1 mod 5).
    # So yes, it is a permutation.
    
    # Check psi(c) = theta(c) - c
    # psi(a, b) = (2a+b-a, b-b) = (a+b, 0).
    # Second component is always 0. Not a permutation.
    
    # We need the second component of psi to vary.
    # So theta_b(a, b) - b must not be constant.
    # So theta_b must depend on a or vary with b non-trivially.
    
    # Let's try theta(a, b) = (a + b, b + a)
    # theta is a permutation?
    # Matrix [[1, 1], [1, 1]] det 0. Not a permutation.
    
    # Let's try theta(a, b) = (a + 2b, 2a + b)
    # Det = 1 - 4 = -3 = 2 mod 5. Invertible.
    # So theta is a permutation of Z5 x Z5? No, Z5 x Z2.
    # The second component is in Z2.
    # theta_b = 2a + b mod 2 = b (since 2a=0).
    # So theta(a, b) = (a + 2b, b).
    # This is the same form as before, just different k, m.
    # psi(a, b) = (a+2b-a, b-b) = (2b, 0).
    # Still 0 in second component.
    
    # It seems any linear map theta(a, b) = (A a + B b, C a + D b)
    # where the map is a permutation of Z5 x Z2.
    # The second component must be a permutation of Z2.
    # So C a + D b mod 2 must be a permutation.
    # This implies C=0, D=1 (or C=1, D=0 etc, but C a mod 2 depends on a).
    # If C=1, D=0: theta_b = a.
    # Then theta(a, b) = (A a + B b, a).
    # Is this a permutation?
    # (a, b) -> (A a + B b, a).
    # Inverse: a' = a, b' = (A a + B b - a) * B^-1?
    # If B is invertible mod 5, yes.
    # So let's try this class.
    # psi(a, b) = theta(a, b) - (a, b) = (A a + B b - a, a - b) = ((A-1)a + B b, a - b).
    # We need psi to be a permutation.
    # Let's pick A=2, B=1.
    # theta(a, b) = (2a + b, a).
    # psi(a, b) = (a + b, a - b).
    # Is (a+b, a-b) a permutation of Z5 x Z2?
    # Let u = a+b, v = a-b.
    # a = (u+v)/2 = (u+v)*3 (mod 5).
    # b = (u-v)/2 = (u-v)*3 (mod 5).
    # Since 2 is invertible mod 5, this is a bijection on Z5 x Z5.
    # But we are in Z5 x Z2.
    # u is in Z5, v is in Z2?
    # No, u = a+b. a in Z5, b in Z2.
    # This addition is not well-defined in Z5 for b.
    # We must keep components separate.
    # psi_a = (A-1)a + B b mod 5.
    # psi_b = a - b mod 2.
    # We need the pair (psi_a, psi_b) to be unique for each (a, b).
    
    # Let's test A=2, B=1.
    # psi_a = a + b (mod 5). Note: b is 0 or 1.
    # psi_b = a - b (mod 2).
    
    # Let's check for collisions.
    # (a, b) and (a', b') give same psi?
    # a + b = a' + b' (mod 5)
    # a - b = a' - b' (mod 2)
    
    # From eq 2: a + b = a' + b' (mod 2).
    # Let S = a+b. S mod 5 is fixed, S mod 2 is fixed.
    # This determines S mod 10? No, a is mod 5, b is mod 2.
    # a is 0..4, b is 0..1.
    # a+b is 0..5.
    # If a+b = a'+b' (mod 5), then a+b = a'+b' or a+b = a'+b'+5.
    
    # Case 1: a+b = a'+b'.
    # Then a-b = a'-b' (mod 2) is automatically satisfied?
    # a+b = a'+b' => a-a' = b'-b.
    # a-b = a'-b' => a-a' = b-b'.
    # So b'-b = b-b' => 2(b-b') = 0. True for b in Z2.
    # So if a+b = a'+b' (mod 5), does it imply (a,b)=(a',b')?
    # Not necessarily.
    # Example: a=0, b=1 => sum=1. a'=1, b'=0 => sum=1.
    # psi(0,1) = (1, 1) [psi_a=1, psi_b=1-1=0? No. a=0, b=1. psi_b = 0-1 = -1 = 1].
    # psi(1,0) = (1, 1) [psi_a=1, psi_b=1-0=1].
    # Collision! (0,1) and (1,0) map to (1,1).
    # So A=2, B=1 fails.
    
    # Let's try A=3, B=1.
    # psi_a = 2a + b.
    # psi_b = a - b.
    # Check collision for (0,1) and (1,0).
    # (0,1): psi_a = 1, psi_b = 1.
    # (1,0): psi_a = 2, psi_b = 1.
    # No collision.
    
    # Let's verify if (3a+b, a-b) is a permutation.
    # We can just code the search for A, B in this class.
    # There are few options.
    
    # If we find one, we construct the squares.
    
    # Let's implement the search for A, B in {0..4} x {0..4}
    # such that theta(a,b) = (A*a + B*b, a) is a permutation (requires B != 0 mod 5)
    # and psi is a permutation.
    
    valid_theta = None
    for A in range(5):
        for B in range(1, 5): # B must be non-zero for theta to be perm
            # Check if psi is permutation
            psi_vals = set()
            is_perm = True
            for a in range(5):
                for b in range(2):
                    # theta(a, b) = (A*a + B*b % 5, a)
                    # Note: theta must be a permutation of G.
                    # We assumed form (A*a+B*b, a).
                    # Is it always a permutation?
                    # (a,b) -> (x, a). x = A*a + B*b.
                    # For fixed a, b varies 0,1. x varies by B.
                    # If B != 0, x is distinct for b=0,1.
                    # For different a, the second component 'a' is distinct.
                    # So yes, it is a permutation.
                    
                    # Calculate psi
                    # psi_a = (A*a + B*b - a) % 5 = ((A-1)*a + B*b) % 5
                    # psi_b = (a - b) % 2
                    pa = ((A-1)*a + B*b) % 5
                    pb = (a - b) % 2
                    key = (pa, pb)
                    if key in psi_vals:
                        is_perm = False
                        break
                    psi_vals.add(key)
            if is_perm:
                valid_theta = (A, B)
                break
        if valid_theta:
            break
            
    if valid_theta:
        A, B = valid_theta
        # Construct L1, L2
        L1 = [[0]*10 for _ in range(10)]
        L2 = [[0]*10 for _ in range(10)]
        for r in range(10):
            ra, rb = to_group(r)
            for c in range(10):
                ca, cb = to_group(c)
                # L1 = r + c
                l1_g = add_g((ra, rb), (ca, cb))
                L1[r][c] = to_int(l1_g)
                
                # L2 = r + theta(c)
                # theta(c) = (A*ca + B*cb % 5, ca)
                # Note: B*cb is in Z5.
                tc_a = (A*ca + B*cb) % 5
                tc_b = ca
                tc_g = (tc_a, tc_b)
                l2_g = add_g((ra, rb), tc_g)
                L2[r][c] = to_int(l2_g)
                
        # Verify
        # (Optional) Check orthogonality
        pairs = set()
        for r in range(10):
            for c in range(10):
                pairs.add((L1[r][c], L2[r][c]))
        if len(pairs) == 100:
            return [L1, L2]
            
    # Fallback: return 1 square
    return [[[(i+j)%10 for j in range(10)] for i in range(10)]]

# EVOLVE-BLOCK-END
