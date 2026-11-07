import math
import time

# EVOLVE-BLOCK-START
def construct():
    # 210 = 2 * 3 * 5 * 7
    # We need to select 1..100 distinct integers x_i such that:
    # 1. 210 < x_i <= 10210
    # 2. 210 * (prod x_i) is a perfect square.
    # 3. Minimize max(x_i) - 210.
    
    # Let x_i = d_i * s_i^2 where d_i is square-free.
    # Then prod(x_i) = (prod d_i) * (prod s_i)^2.
    # Condition: 210 * (prod d_i) * (prod s_i)^2 is a square.
    # This implies 210 * (prod d_i) must be a square.
    # Let 210 * (prod d_i) = k^2.
    # Since 210 = 2*3*5*7, and d_i are square-free, the product of d_i must
    # contain each prime factor of 210 an odd number of times to make the
    # total exponent even in 210 * prod(d_i).
    # Wait, 210 has exponents 1 for all four primes.
    # So prod(d_i) must have exponents 1 mod 2 for each of 2,3,5,7?
    # Let e_p be the exponent of prime p in prod(d_i).
    # Exponent of p in 210 * prod(d_i) is 1 + e_p.
    # We need 1 + e_p to be even => e_p is odd.
    # So each prime factor of 210 must appear in an odd number of the d_i's.
    # Since d_i are square-free, the prime p appears in d_i iff p divides d_i.
    # So for each p in {2,3,5,7}, the number of d_i divisible by p must be odd.
    
    # To minimize max(x_i), we want to minimize max(d_i * s_i^2).
    # s_i must be such that x_i > 210. So s_i > sqrt(210/d_i).
    # s_i = floor(sqrt(210/d_i)) + 1.
    
    # We can partition the set of primes {2,3,5,7} into groups.
    # Each group defines a d_i which is the product of primes in that group.
    # However, the condition is on the parity of the count of d_i containing p.
    # This is equivalent to saying that the vector of incidence of primes in d_i's
    # must sum to 1 mod 2 for each prime.
    # The simplest solution is to use d_i = p for each p in {2,3,5,7}.
    # Then we have 4 numbers.
    # d_1=2, d_2=3, d_3=5, d_4=7.
    # s_1 = min s s.t. 2*s^2 > 210 => s^2 > 105 => s=11 (121*2=242)
    # s_2 = min s s.t. 3*s^2 > 210 => s^2 > 70 => s=9 (81*3=243)
    # s_3 = min s s.t. 5*s^2 > 210 => s^2 > 42 => s=7 (49*5=245)
    # s_4 = min s s.t. 7*s^2 > 210 => s^2 > 30 => s=6 (36*7=252)
    # Max is 252. Raw value 42.
    
    # Can we do better?
    # If we merge two primes into one d_i, say d_1 = 2*3=6.
    # Then we need the remaining primes 5,7 to be covered by other d_j's such that
    # the parity condition holds.
    # If we use d_1=6, d_2=5, d_3=7.
    # Check parity:
    # p=2: in d_1 (1) -> odd. OK.
    # p=3: in d_1 (1) -> odd. OK.
    # p=5: in d_2 (1) -> odd. OK.
    # p=7: in d_3 (1) -> odd. OK.
    # So {6, 5, 7} is a valid set of kernels.
    # Calculate numbers:
    # d=6: s^2 > 210/6 = 35 => s=6 (36*6=216)
    # d=5: s^2 > 210/5 = 42 => s=7 (49*5=245)
    # d=7: s^2 > 210/7 = 30 => s=6 (36*7=252)
    # Max is 252. Same as before.
    
    # Try d_1=10 (2*5), d_2=3, d_3=7.
    # d=10: s^2 > 21 => s=5 (25*10=250)
    # d=3: s^2 > 70 => s=9 (81*3=243)
    # d=7: s^2 > 30 => s=6 (36*7=252)
    # Max 252.
    
    # Try d_1=14 (2*7), d_2=3, d_3=5.
    # d=14: s^2 > 15 => s=4 (16*14=224)
    # d=3: s^2 > 70 => s=9 (243)
    # d=5: s^2 > 42 => s=7 (245)
    # Max 245. Raw 35. Better!
    
    # Try d_1=15 (3*5), d_2=2, d_3=7.
    # d=15: s^2 > 14 => s=4 (16*15=240)
    # d=2: s^2 > 105 => s=11 (242)
    # d=7: s^2 > 30 => s=6 (252)
    # Max 252.
    
    # Try d_1=21 (3*7), d_2=2, d_3=5.
    # d=21: s^2 > 10 => s=4 (16*21=336) -> too big?
    # s=4 => 336.
    # d=2: 242.
    # d=5: 245.
    # Max 336. Worse.
    
    # Try d_1=30 (2*3*5), d_2=7.
    # d=30: s^2 > 7 => s=3 (9*30=270)
    # d=7: 252.
    # Max 270.
    
    # Try d_1=42 (2*3*7), d_2=5.
    # d=42: s^2 > 5 => s=3 (9*42=378)
    # d=5: 245.
    # Max 378.
    
    # Try d_1=35 (5*7), d_2=2, d_3=3.
    # d=35: s^2 > 6 => s=3 (9*35=315)
    # d=2: 242.
    # d=3: 243.
    # Max 315.
    
    # Try d_1=6, d_2=10, d_3=? 
    # If we use more numbers, can we reduce the max?
    # The max is dominated by the smallest d_i requiring a large s_i.
    # s_i ~ sqrt(210/d_i).
    # x_i = d_i * (sqrt(210/d_i))^2 = 210.
    # So x_i is just above 210.
    # The max is determined by the d_i that makes x_i largest.
    # x_i = d_i * (floor(sqrt(210/d_i)) + 1)^2.
    # Let's check all possible sets of square-free d_i formed from subsets of {2,3,5,7}
    # such that each prime appears in an odd number of d_i's.
    # This is a linear algebra problem over GF(2).
    # But the number of primes is small.
    
    # Let's just hardcode the best found:
    # d = [14, 3, 5] -> [224, 243, 245] -> Max 245.
    # Is there a 4-number solution better than 245?
    # We had 252.
    # What about 3 numbers with different d's?
    # We checked {14,3,5} -> 245.
    # What about {6,5,7} -> 252.
    # What about {10,3,7} -> 252.
    # What about {15,2,7} -> 252.
    # What about {21,2,5} -> 336.
    # What about {35,2,3} -> 315.
    # What about {30,7} -> 270.
    # What about {42,5} -> 378.
    # What about {70,3} -> 315 (d=70, s^2>3, s=2, 4*70=280; d=3, 243. Max 280).
    # d=70: 210/70=3, s=2, x=280.
    # d=3: 243.
    # Max 280.
    # What about {105,2}?
    # d=105: 210/105=2, s=2, x=420.
    # d=2: 242.
    # Max 420.
    
    # So 245 seems best among 3 numbers.
    # What about 4 numbers?
    # We need 4 square-free d_i from {2,3,5,7} subsets such that parity is odd.
    # If we use single primes: {2,3,5,7} -> 252.
    # Can we mix?
    # e.g. {2, 3, 5, 7} is the only way to have 4 sets if we assume disjoint?
    # No, they don't have to be disjoint.
    # But if they are not disjoint, a prime might appear in multiple d_i.
    # Example: d_1=2, d_2=2*3=6? No, d_i must be square-free.
    # If d_1=2 and d_2=6, then p=2 appears in both.
    # Count for 2: 2 (even) -> Bad.
    # So if a prime appears in multiple d_i, it must appear in an odd total count.
    # With 4 sets, it's hard to improve on 252 because the "single prime" case
    # balances the s_i values well.
    # Actually, let's re-verify 245.
    # 224 = 14 * 16 = 14 * 4^2.
    # 243 = 3 * 81 = 3 * 9^2.
    # 245 = 5 * 49 = 5 * 7^2.
    # Product = 14*3*5 * 4^2 * 9^2 * 7^2 = 210 * (4*9*7)^2 = 210 * 252^2.
    # 210 * (210 * 252^2) = 210^2 * 252^2 = (210*252)^2. Square!
    # Distinct: 224, 243, 245. Yes.
    # Max 245. Raw 35.
    
    return [224, 243, 245]
# EVOLVE-BLOCK-END
