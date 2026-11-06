        power = p2
        next_power = power * p
        while power <= N_MAX:
            # check m % next_power != 0
            # multiply val[m] by power
            power = next_power
            next_power *= p
