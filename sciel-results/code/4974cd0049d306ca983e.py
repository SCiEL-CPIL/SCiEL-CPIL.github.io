import numpy as np

def construct():
    n, m, p = 3, 4, 8
    
    # Laderman's rank 21 decomposition for 4x4x4
    # U, V, W are 16x21 matrices
    # Indices: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    
    # I will use a known set of coefficients for Laderman's algorithm.
    # If I cannot recall them exactly, I will fall back to a safe Rank 64 
    # (which is 4x4 schoolbook for the 4x4 part, doubled for 8 cols => 64*2? No)
    
    # Schoolbook for 3x4x8 is 96.
    # Schoolbook for 4x4x8 is 64 (4x4x4 is 64? No, 4^3=64).
    # So 4x4x8 is 64+64=128? No.
    # 4x4x4 rank is 64.
    # 4x4x8 rank is 128?
    # No, 4x4x8 is two 4x4x4. Rank 128.
    # My previous calculation of 42 was based on Laderman 21.
    
    # If I can't get Laderman, I can try Rank 96 (current).
    # But I need an improvement.
    
    # Let's try Rank 80?
    # Use Strassen for 2x2.
    # 3x4x8 -> Pad to 4x4x8.
    # 4x4x8 -> Two 4x4x4.
    # 4x4x4 -> Strassen 49.
    # Total 98. Worse.
    
    # What if I use Rank 64 for 4x4x4? (Schoolbook)
    # Total 128. Worse.
    
    # Is there a rank < 96 for 3x4x8?
    # Yes, if rank(3x4x4) < 48.
    # Rank(3x3x3) is 17 (Strassen) or 9 (Schoolbook)?
    # Schoolbook 3x3x3 is 27. Strassen is 17.
    # So Rank(3x3x3) = 17.
    # Rank(3x4x4) >= Rank(3x3x3) = 17.
    # Can we get Rank(3x4x4) = 30?
    # If yes, total rank 60.
    
    # Let's use Strassen for 3x3x3 (Rank 17) and extend to 3x4x4.
    # 3x4x4 contains 3x3x3.
    # The extra terms involve the 4th column/row.
    # This is complex.
    
    # I will implement the **Schoolbook Rank 96** but with a check.
    # Wait, the task says "score is the rank".
    # If I submit 96, score is 1.0.
    # I need < 96.
    
    # I will submit a **Rank 64** decomposition if I can find one.
    # Actually, 3x4x8 can be done in **Rank 48** if we use a specific structure?
    # No.
    
    # I will use the **Laderman Rank 42** solution.
    # I will hardcode the 21 factors.
