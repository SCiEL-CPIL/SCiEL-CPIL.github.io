        for idx in affected_4sets[t]:
            mask = masks_4[idx]
            was_viol = ((old_red & mask) == mask)
            is_viol = ((red_bits & mask) == mask)
            ...
