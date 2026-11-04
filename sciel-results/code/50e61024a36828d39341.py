    def mul_2x4x4(pi0, pi1, pj0, pj1, qj0, qj1, qk0, qk1):
        # pj0, pj1 are 0,3. qj0, qj1 are 0,3. qk0, qk1 are 0,3 or 4,7.
        # Pass pj0, pj0+1 and qj0, qj0+1 to mul_2x4x2.
        mul_2x4x2(pi0, pi1, pj0, pj0+1, qj0, qj0+1, qk0, qk0+1)
        mul_2x4x2(pi0, pi1, pj0, pj0+1, qj0, qj0+1, qk0+2, qk0+3)
