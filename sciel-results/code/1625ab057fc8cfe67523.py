import time

def construct():
    # Theoretical minimum for n=12 is 1025.
    # Sequence: 1, 2, 4, ..., 512 (10 elements, sum 1023)
    # a_10 = 513 (must be > 511 and distinct from 512)
    # S_10 = 1023 + 513 = 1536? No.
    # S_9 = 511.
    # a_10 = 513.
    # S_10 = 511 + 513 = 1024.
    # Wait, S_9 is sum of first 9?
    # a_0..a_8 are 1..256. Sum = 511.
    # a_9 = 512.
    # S_9 = 511 + 512 = 1023.
    # a_10 must be > 1023.
    # Min a_10 = 1024.
    # But a_10 must be distinct. 1024 is not in {1..512}.
    # So a_10 = 1024 is valid.
    # S_10 = 1023 + 1024 = 2047.
    # a_11 must be > 2047.
    # Min a_11 = 2048.
    
    # My previous derivation:
    # a_0..a_8 = 1..256. S_8 = 511.
    # a_9 = 512. S_9 = 1023.
    # a_10 > 1023. Min 1024.
    # a_10 = 1024.
    # S_10 = 2047.
    # a_11 > 2047. Min 2048.
    
    # Where did I get 1025?
    # Ah, I grouped a_0..a_9 as 1..512.
    # That is 10 numbers.
    # a_0=1, a_1=2, a_2=4, a_3=8, a_4=16, a_5=32, a_6=64, a_7=128, a_8=256, a_9=512.
    # Sum S_9 = 1023.
    # Next is a_10.
    # a_10 > S_9 = 1023.
    # Min a_10 = 1024.
    # Is 1024 distinct? Yes.
    # So a_10 = 1024.
    # S_10 = 1023 + 1024 = 2047.
    # Next is a_11.
    # a_11 > S_10 = 2047.
    # Min a_11 = 2048.
    
    # So the minimum is 2048?
    # Let's re-read the distinctness condition.
    # "Every subset... different sum".
    # This is satisfied by superincreasing.
    # Is it possible to have non-superincreasing with smaller max?
    
    # Let's re-evaluate the 1025 case.
    # I had: a_0..a_8 = 1..256 (9 numbers). S_8 = 511.
    # a_9 = 512. S_9 = 1023.
    # a_10 = 513?
    # Condition: A (sums of a_0..a_9) and A+a_10 disjoint.
    # A = [0, 1023].
    # If a_10 = 513, A+513 = [513, 1536].
    # Intersection [513, 1023] is NOT empty.
    # So a_10 CANNOT be 513.
    # a_10 must be > 1023.
    # So a_10 = 1024.
    
    # So the minimum is indeed 2048.
    # The seed program is already optimal?
    
    # Wait, the task says "Minimize the largest integer".
    # Seed returns 2048.
    # If 2048 is the minimum, score is 1.0.
    
    # Let's check if I can break the superincreasing property.
    # If I use a set that is NOT superincreasing, can I get a smaller max?
    # Example: n=3.
    # Superincreasing: 1, 2, 4. Max 4. Sums: 0,1,2,3,4,5,6,7.
    # Can we do max 3?
    # 1, 2, 3. Sums: 0,1,2,3,3,4,5,6. Duplicate 3.
    # 1, 1, 3? Distinct integers required.
    # 1, 2, 3 fails.
    # So 4 is min for n=3.
    
    # n=4.
    # Superincreasing: 1, 2, 4, 8. Max 8.
    # Can we do 7?
    # 1, 2, 4, 7.
    # Sums of 1,2,4: 0..7.
    # Add 7: 7..14.
    # Intersection at 7.
    # 1, 2, 4, 6?
    # Sums 1,2,4: 0..7.
    # Add 6: 6..13.
    # Intersection 6,7.
    # 1, 2, 3, 7?
    # 1,2,3 sums: 0,1,2,3,3,4,5,6. Duplicates.
    
    # It seems superincreasing is optimal.
    
    # However, there is a known sequence: Mian-Chowla sequence?
    # No, that's for distinct sums of pairs.
    
    # What about "Sidon sets"?
    
    # Let's check n=2.
    # 1, 2. Max 2.
    # 1, 3. Max 3.
    # Min 2.
    
    # n=1.
    # 1.
    
    # It appears the powers of 2 are the unique minimal solution.
    
    # But wait.
    # What if we don't start with 1?
    # If we start with 2, 3, 5, 9...
    # Sum of 2,3 is 5. Next > 5 is 6.
    # 2, 3, 6. Max 6.
    # Powers of 2: 1, 2, 4. Max 4.
    # 4 < 6.
    
    # So starting with 1 is best.
    
    # Conclusion: The seed program is likely optimal.
    # But I must provide a program.
    # I will provide the powers of 2.
    
    return [1 << i for i in range(12)]
