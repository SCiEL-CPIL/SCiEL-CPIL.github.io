import time
import numpy as np

def construct():
    """
    Strategy: Use roots of unity to construct a polynomial P(x) such that P(x)^2 has minimal nonzero terms.
    
    Consider P(x) = sum_{k=0}^{11} x^{k} * c_k.
    If we choose coefficients c_k such that the polynomial is related to a cyclotomic polynomial or a product of terms (1 - x^{d_i}), 
    the square might have cancellations.
    
    A simpler, more direct approach for "Sparse squares":
    Recall that (a - b)^2 = a^2 - 2ab + b^2 has 3 terms.
    (a + b - c)^2 = a^2 + b^2 + c^2 + 2ab - 2ac - 2bc has up to 6 terms.
    
    Let's look at P(x) = 1 - x - x^2 - ... - x^{11}.
    P(x) = 1 - (x^{12}-1)/(x-1) = (x^{12}-1 - (x-1))/(x-1) = (x^{12}-x)/(x-1) ... wait.
    1 - x - ... - x^{11} = 1 - (x^{12}-x)/(x-1) = ((x-1) - (x^{12}-x))/(x-1) = (2x - 1 - x^{12})/(x-1).
    
    Let's try a known identity.
    (1 - x - x^2)^2 = 1 - 2x - x^2 + 2x^3 + x^4. (5 terms)
    Input has 3 terms.
    
    We need 12 input terms.
    Let's try P(x) = (1 - x - x^2)^4.
    Degree 8. Number of terms in P?
    (1 - x - x^2)^2 = 1 - 2x - x^2 + 2x^3 + x^4.
    (1 - x - x^2)^4 = (1 - 2x - x^2 + 2x^3 + x^4)^2.
    This will have many terms.
    
    Let's try a product of disjoint blocks.
    P(x) = (1 - x)^6 (1 - x^2)^3 ? No, coefficients must be integers.
    
    Let's look at P(x) = \prod_{i=1}^6 (1 - x^{a_i}).
    If a_i are distinct powers of 2, P(x) has 2^6 = 64 terms. Too many.
    We need exactly 12 nonzero terms in P(x).
    
    How to get exactly 12 terms?
    Maybe P(x) = A(x) * B(x) where A and B have few terms.
    If A has 3 terms and B has 4 terms, P has up to 12 terms (if no collisions).
    Let A(x) = 1 - x - x^2.
    Let B(x) = 1 - x - x^3 - x^4.
    P(x) = (1 - x - x^2)(1 - x - x^3 - x^4)
    = 1 - x - x^3 - x^4 - x + x^2 + x^4 + x^5 - x^2 + x^3 + x^5 + x^6
    = 1 - 2x + (x^2 - x^2) + (-x^3 + x^3) + (-x^4 + x^4) + 2x^5 + x^6
    = 1 - 2x + 2x^5 + x^6.
    This has 4 terms. We need 12.
    
    Let's try to make the product have NO internal cancellations to preserve term count, or specific ones.
    If we pick A and B such that their exponents are "independent" (e.g. A uses even powers, B uses odd powers, or separated ranges), the term count is product of term counts.
    3 * 4 = 12.
    
    Let A(x) = 1 + x^{20} + x^{40}. (3 terms)
    Let B(x) = 1 - x - x^2 - x^3. (4 terms)
    P(x) = A(x)B(x) = (1 + x^{20} + x^{40})(1 - x - x^2 - x^3)
    Terms in P:
    1, -1, -1, -1 (from 1*B)
    x^{20}, -x^{21}, -x^{22}, -x^{23} (from x^{20}*B)
    x^{40}, -x^{41}, -x^{42}, -x^{43} (from x^{40}*B)
    Total 12 terms. Coeffs are +/-1.
    Max degree 43. Constraint: length <= 41 (degree <= 40).
    So we need max degree <= 40.
    
    Adjust exponents to fit in degree <= 40.
    Let A(x) = 1 + x^{15} + x^{30}.
    Let B(x) = 1 - x - x^2 - x^3.
    P(x) = (1 + x^{15} + x^{30})(1 - x - x^2 - x^3)
    Max degree = 30 + 3 = 33.
    Terms:
    1, -1, -1, -1
    x^{15}, -x^{16}, -x^{17}, -x^{18}
    x^{30}, -x^{31}, -x^{32}, -x^{33}
    All distinct. 12 terms.
    Coefficients are in [-1, 1] which is within [-64, 64].
    
    Now, what is P(x)^2?
    P(x) = B(x) + x^{15}B(x) + x^{30}B(x).
    P(x)^2 = B(x)^2 + x^{30}B(x)^2 + x^{60}B(x)^2 + 2x^{15}B(x)^2 + 2x^{30}B(x)^2 + 2x^{45}B(x)^2.
    
    We need to calculate the number of nonzero terms in this sum.
    Let Q(x) = B(x)^2 = (1 - x - x^2 - x^3)^2.
    Q(x) = (1 - x - x^2 - x^3)(1 - x - x^2 - x^3)
    = 1 - x - x^2 - x^3 - x + x^2 + x^3 + x^4 - x^2 + x^3 + x^4 + x^5 - x^3 + x^4 + x^5 + x^6
    = 1 - 2x + (-1+1-1+1)x^2 + (-1+1+1-1)x^3 + (1+1+1)x^4 + (1+1)x^5 + x^6
    Let's re-calculate carefully.
    Coeff of x^0: 1*1 = 1
    Coeff of x^1: 1*(-1) + (-1)*1 = -2
    Coeff of x^2: 1*(-1) + (-1)*(-1) + (-1)*1 = -1 + 1 - 1 = -1
    Coeff of x^3: 1*(-1) + (-1)*(-1) + (-1)*(-1) + (-1)*1 = -1 + 1 + 1 - 1 = 0
    Coeff of x^4: (-1)*(-1) + (-1)*(-1) + (-1)*1 = 1 + 1 - 1 = 1
    Coeff of x^5: (-1)*(-1) + (-1)*1 = 1 - 1 = 0
    Coeff of x^6: (-1)*(-1) = 1
    
    So Q(x) = 1 - 2x - x^2 + x^4 + x^6.
    Number of nonzero terms in Q(x) is 5.
    
    Now P(x)^2 = Q(x) + 2x^{15}Q(x) + 3x^{30}Q(x) + 2x^{45}Q(x) + x^{60}Q(x).
    Wait, the cross terms were:
    2 * (1) * (x^{15}B) = 2x^{15} B^2 = 2x^{15} Q
    2 * (1) * (x^{30}B) = 2x^{30} B^2 = 2x^{30} Q
    2 * (x^{15}B) * (x^{30}B) = 2x^{45} B^2 = 2x^{45} Q
    And the squares:
    B^2 = Q
    (x^{15}B)^2 = x^{30} B^2 = x^{30} Q
    (x^{30}B)^2 = x^{60} B^2 = x^{60} Q
    
    So P(x)^2 = Q(x) + 2x^{15}Q(x) + (1+2)x^{30}Q(x) + 2x^{45}Q(x) + x^{60}Q(x)
    P(x)^2 = Q(x) + 2x^{15}Q(x) + 3x^{30}Q(x) + 2x^{45}Q(x) + x^{60}Q(x)
    
    We need to check for collisions between the shifted versions of Q(x).
    Q(x) has support at {0, 1, 2, 4, 6}.
    Shifts are by 0, 15, 30, 45, 60.
    Supports:
    S0 = {0, 1, 2, 4, 6}
    S15 = {15, 16, 17, 19, 21}
    S30 = {30, 31, 32, 34, 36}
    S45 = {45, 46, 47, 49, 51}
    S60 = {60, 61, 62, 64, 66}
    
    Are these sets disjoint?
    Max of S0 is 6. Min of S15 is 15. Gap.
    Max of S15 is 21. Min of S30 is 30. Gap.
    Max of S30 is 36. Min of S45 is 45. Gap.
    Max of S45 is 51. Min of S60 is 60. Gap.
    
    Yes, they are completely disjoint.
    Therefore, the number of nonzero terms in P(x)^2 is simply 5 * 5 = 25.
    
    Can we do better?
    The seed program likely returns something worse or similar.
    Is it possible to get fewer than 25?
    The minimum number of terms in the square of a 12-term polynomial is at least 12 (if it were a monomial squared, but it's not).
    Actually, if P(x) = sum c_i x^{e_i}, P(x)^2 = sum c_i^2 x^{2e_i} + sum_{i!=j} c_i c_j x^{e_i+e_j}.
    If all e_i+e_j are distinct and c_i c_j != 0, we have 12 + 132 = 144 terms.
    We want cancellations.
    
    My construction yields 25 terms.
    
    Let's check if we can reduce the number of terms in Q(x).
    Q(x) = B(x)^2. B has 4 terms.
    Can we find a 4-term polynomial B(x) with integer coeffs such that B(x)^2 has fewer than 5 terms?
    B(x) = 1 - x - x^2 - x^3 gave 5.
    Try B(x) = 1 - 2x + x^2 = (1-x)^2.
    B(x)^2 = (1-x)^4 = 1 - 4x + 6x^2 - 4x^3 + x^4. 5 terms.
    Try B(x) = 1 - x^2. 2 terms. B^2 = 1 - 2x^2 + x^4. 3 terms.
    But we need B to have 4 terms to make P have 12 terms (3 * 4 = 12).
    If B has 2 terms, A must have 6 terms.
    Let A(x) have 6 terms, B(x) have 2 terms.
    B(x) = 1 - x^2. B^2 = 1 - 2x^2 + x^4 (3 terms).
    Let A(x) = 1 + x^k + x^{2k} + x^{3k} + x^{4k} + x^{5k} (geometric series).
    A(x) = (1 - x^{6k}) / (1 - x^k).
    P(x) = A(x) B(x).
    If we choose k large enough, the terms in A(x) and shifts don't collide with B(x) structure?
    Actually, P(x) = (1 + x^k + ... + x^{5k})(1 - x^2).
    Terms in P:
    1, -x^2
    x^k, -x^{k+2}
    ...
    x^{5k}, -x^{5k+2}
    If k > 2, these are all distinct. Total 12 terms.
    
    Now P(x)^2 = A(x)^2 B(x)^2.
    B(x)^2 = 1 - 2x^2 + x^4.
    A(x)^2 = (1 + x^k + ... + x^{5k})^2.
    A(x) is a geometric sum.
    A(x)^2 = sum_{i=0}^5 sum_{j=0}^5 x^{(i+j)k} = sum_{m=0}^{10} (m+1 if m<=5 else 11-m) x^{mk}.
    Coefficients of A(x)^2:
    m=0: 1
    m=1: 2
    m=2: 3
    m=3: 4
    m=4: 5
    m=5: 6
    m=6: 5
    m=7: 4
    m=8: 3
    m=9: 2
    m=10: 1
    All nonzero. So A(x)^2 has 11 terms.
    
    P(x)^2 = A(x)^2 (1 - 2x^2 + x^4).
    Let R(x) = A(x)^2. Support of R is {0, k, 2k, ..., 10k}.
    P(x)^2 = R(x) - 2x^2 R(x) + x^4 R(x).
    
    We need to check for collisions between R(x), x^2 R(x), and x^4 R(x).
    If k is large enough such that the gaps between multiples of k are > 4, then the supports are disjoint.
    Support of R: {0, k, 2k, ..., 10k}.
    Support of x^2 R: {2, k+2, 2k+2, ..., 10k+2}.
    Support of x^4 R: {4, k+4, 2k+4, ..., 10k+4}.
    
    If k > 4, then for any i, i*k, i*k+2, i*k+4 are distinct.
    Also, (i+1)k > i*k + 4 if k > 4.
    So all supports are disjoint.
    
    Number of terms in P(x)^2 = 11 (from R) + 11 (from -2x^2 R) + 11 (from x^4 R) = 33 terms.
    
    33 is worse than 25.
    
    So the first construction (A with 3 terms, B with 4 terms) giving 25 terms is better.
    
    Can we improve the 25 terms?
    We had Q(x) = B(x)^2 with 5 terms.
    Can we find B(x) with 4 terms such that B(x)^2 has 4 terms?
    B(x) = c0 + c1 x + c2 x^2 + c3 x^3.
    B(x)^2 = c0^2 + 2c0c1 x + (c1^2 + 2c0c2) x^2 + (2c0c3 + 2c1c2) x^3 + (c2^2 + 2c1c3) x^4 + 2c2c3 x^5 + c3^2 x^6.
    We want 3 of these coefficients to be zero.
    
    Try B(x) = 1 - x - x^2 + x^3.
    c0=1, c1=-1, c2=-1, c3=1.
    Coeffs of B^2:
    x^0: 1
    x^1: -2
    x^2: 1 - 2 = -1
    x^3: 2 - 2 = 0
    x^4: 1 - 2 = -1
    x^5: -2
    x^6: 1
    Terms: 1, -2, -1, 0, -1, -2, 1.
    Nonzero: 6 terms. Worse.
    
    Try B(x) = 1 - x + x^2 - x^3.
    c0=1, c1=-1, c2=1, c3=-1.
    x^0: 1
    x^1: -2
    x^2: 1 + 2 = 3
    x^3: -2 + 2 = 0
    x^4: 1 - 2 = -1
    x^5: -2
    x^6: 1
    Nonzero: 6 terms.
    
    Try B(x) = 1 + x - x^2 - x^3.
    c0=1, c1=1, c2=-1, c3=-1.
    x^0: 1
    x^1: 2
    x^2: 1 - 2 = -1
    x^3: -2 - 2 = -4
    x^4: 1 - 2 = -1
    x^5: -2
    x^6: 1
    Nonzero: 7 terms.
    
    It seems hard to get B(x)^2 below 5 terms with 4-term B.
    The minimum number of terms in the square of a polynomial with n nonzero terms is at least n?
    No, (1-x)^2 = 1-2x+x^2 (3 terms from 2).
    (1-x-x^2)^2 = 5 terms from 3.
    (1-x-x^2-x^3)^2 = 5 terms from 4.
    (1-x-x^2-x^3-x^4)^2?
    Coeffs:
    1, -2, -1, 0, 1, -2, 1?
    Let's check B(x) = 1 - x - x^2 - x^3 - x^4.
    x^0: 1
    x^1: -2
    x^2: 1 - 2 = -1
    x^3: -1 + 2 - 2 = -1
    x^4: -1 + 2 - 2 + 1 = 0?
    Sum c_i c_{4-i}:
    c0c4 + c1c3 + c2c2 + c3c1 + c4c0 = -1 + 1 - 1 + 1 - 1 = -1.
    Let's re-calculate B(x) = 1 - x - x^2 - x^3 - x^4.
    c = [1, -1, -1, -1, -1].
    S0 = 1
    S1 = -2
    S2 = 1 + 2 - 1 = 2?
    S2 = c0c2 + c1c1 + c2c0 = -1 + 1 - 1 = -1.
    S3 = c0c3 + c1c2 + c2c1 + c3c0 = -1 + 1 - 1 - 1 = -2.
    S4 = c0c4 + c1c3 + c2c2 + c3c1 + c4c0 = -1 + 1 - 1 + 1 - 1 = -1.
    S5 = c1c4 + c2c3 + c3c2 + c4c1 = 1 + 1 + 1 + 1 = 4.
    S6 = c2c4 + c3c3 + c4c2 = 1 + 1 + 1 = 3.
    S7 = c3c4 + c4c3 = 1 + 1 = 2.
    S8 = c4c4 = 1.
    All nonzero. 9 terms.
    
    So 5 terms for B^2 seems minimal for 4-term B.
    
    Thus, 25 terms for P^2 seems a strong candidate.
    
    Let's verify the constraints for the 25-term solution.
    P(x) = (1 + x^{15} + x^{30})(1 - x - x^2 - x^3).
    Coeffs of P:
    Indices: 0, 1, 2, 3, 15, 16, 17, 18, 30, 31, 32, 33.
    Values:
    0: 1
    1: -1
    2: -1
    3: -1
    15: 1
    16: -1
    17: -1
    18: -1
    30: 1
    31: -1
    32: -1
    33: -1
    
    Length: 34 (indices 0 to 33). Max index 33 < 41. OK.
    Nonzero terms: 12. OK.
    Coeffs in [-64, 64]. OK.
    Final coefficient (at 33) is -1 (nonzero). OK.
    
    Square has 25 terms.
    
    Can we do better by overlapping the shifts?
    If we choose shifts such that Q(x) terms cancel?
    Q(x) = 1 - 2x - x^2 + x^4 + x^6.
    P(x)^2 = Q(x) + 2x^s Q(x) + 3x^{2s} Q(x) + 2x^{3s} Q(x) + x^{4s} Q(x).
    If we pick s such that supports overlap, we might get cancellations.
    However, the coefficients are positive in the structure (1, 2, 3, 2, 1) times Q.
    If Q has negative coefficients, and we overlap, we might cancel.
    Q has coeffs: 1, -2, -1, 0, 1, 0, 1.
    
    Let's try s=1.
    P(x) = (1 + x + x^2 + x^3 + ...?) No, A must have 3 terms.
    If s=1, A(x) = 1 + x + x^2.
    B(x) = 1 - x - x^2 - x^3.
    P(x) = (1+x+x^2)(1-x-x^2-x^3).
    P(x) = 1 - x - x^2 - x^3 + x - x^2 - x^3 - x^4 + x^2 - x^3 - x^4 - x^5
    = 1 + 0x + (-1-1+1)x^2 + (-1-1-1)x^3 + (-1-1)x^4 - x^5
    = 1 - x^2 - 3x^3 - 2x^4 - x^5.
    Terms: 1, 0, -1, -3, -2, -1.
    Nonzero terms: 5.
    We need 12 nonzero terms in P. This construction fails the "exactly 12 nonzero terms" constraint.
    
    We need A and B such that the product has exactly 12 nonzero terms.
    This requires that the convolution of A and B does not result in zero coefficients for the 12 potential positions, or that we carefully design them to have 12 nonzeros.
    
    In my first construction, the supports were disjoint, so term count was exactly 3*4=12.
    If we allow overlaps, we risk reducing the term count below 12 or creating zeros.
    To maintain exactly 12 terms, disjoint supports of the "blocks" in A and B is the safest way to guarantee 12 terms.
    Disjoint supports imply disjoint supports in the square (if shifts are large enough), leading to additive term counts.
    
    So 25 terms is a robust result.
    
    Is there a 3-term B with 4-term A?
    B(x) = 1 - x - x^2. B^2 = 1 - 2x - x^2 + 2x^3 + x^4 (5 terms).
    A(x) = 1 + x^k + x^{2k} + x^{3k} (4 terms).
    P(x) = A(x) B(x).
    If k large, 12 terms.
    P(x)^2 = A(x)^2 B(x)^2.
    A(x)^2 = (1+x^k+x^{2k}+x^{3k})^2.
    Coeffs of A^2:
    1, 2, 3, 4, 3, 2, 1 at powers 0, k, 2k, 3k, 4k, 5k, 6k.
    7 terms.
    B^2 has 5 terms.
    Total terms in P^2 = 7 * 5 = 35 (if disjoint).
    35 > 25.
    
    So 3-term A, 4-term B is better.
    
    What if A has 2 terms and B has 6 terms?
    B(x) with 6 terms. Min terms in B^2?
    If B is geometric 1+x+...+x^5, B^2 has 11 terms.
    A(x) = 1 + x^k. A^2 = 1 + 2x^k + x^{2k} (3 terms).
    Total 3 * 11 = 33 terms.
    
    So 25 seems the best simple product construction.
    
    Let's double check if we can get 24 or less.
    Maybe a non-product polynomial?
    The seed program uses local search. It might find something better if it searches long enough, but 25 is a very low number for a 12-term polynomial square.
    
    I will return the polynomial with 25 square terms.
    
    Polynomial:
    coeffs = [0] * 41
    coeffs[0] = 1
    coeffs[1] = -1
    coeffs[2] = -1
    coeffs[3] = -1
    coeffs[15] = 1
    coeffs[16] = -1
    coeffs[17] = -1
    coeffs[18] = -1
    coeffs[30] = 1
    coeffs[31] = -1
    coeffs[32] = -1
    coeffs[33] = -1
    
    Return coeffs.
    """
    coeffs = [0] * 41
    # A(x) = 1 + x^15 + x^30
    # B(x) = 1 - x - x^2 - x^3
    # P(x) = A(x)B(x)
    
    # B part at 0
    coeffs[0] = 1
    coeffs[1] = -1
    coeffs[2] = -1
    coeffs[3] = -1
    
    # B part at 15
    coeffs[15] = 1
    coeffs[16] = -1
    coeffs[17] = -1
    coeffs[18] = -1
    
    # B part at 30
    coeffs[30] = 1
    coeffs[31] = -1
    coeffs[32] = -1
    coeffs[33] = -1
    
    return coeffs

